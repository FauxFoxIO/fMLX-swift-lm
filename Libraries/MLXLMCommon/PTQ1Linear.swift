// Copyright © 2026 Faux Fox.

import Foundation
import MLX
import MLXNN

/// A compact 1.75-bit ternary projection using GGML's PTQ1_0 block layout.
///
/// Each 128-weight block remains in its native 28-byte representation: 26 bytes of
/// base-three codes followed by one little-endian FP16 scale. The Metal kernel decodes
/// trits while accumulating, so inference never materializes an affine 2-bit copy.
open class PTQ1Linear: Linear {
    public static let blockWidth = 128
    public static let bytesPerBlock = 28

    public let inputDimensions: Int
    public let outputDimensions: Int

    public override var shape: (Int, Int) { (outputDimensions, inputDimensions) }

    public init(inputDimensions: Int, outputDimensions: Int) {
        precondition(inputDimensions > 0 && inputDimensions.isMultiple(of: Self.blockWidth))
        precondition(outputDimensions > 0)
        self.inputDimensions = inputDimensions
        self.outputDimensions = outputDimensions
        let weight = MLXArray.zeros(
            [outputDimensions, inputDimensions / Self.blockWidth, Self.bytesPerBlock],
            dtype: .uint8)
        super.init(weight: weight)
    }

    public init(weight: MLXArray, inputDimensions: Int, outputDimensions: Int) {
        precondition(inputDimensions > 0 && inputDimensions.isMultiple(of: Self.blockWidth))
        precondition(outputDimensions > 0)
        precondition(
            weight.dtype == .uint8
                && weight.shape
                    == [outputDimensions, inputDimensions / Self.blockWidth, Self.bytesPerBlock],
            "Invalid PTQ1_0 weight tensor")
        self.inputDimensions = inputDimensions
        self.outputDimensions = outputDimensions
        super.init(weight: weight)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        precondition(x.dim(-1) == inputDimensions, "PTQ1_0 input dimension mismatch")
        let rows = x.size / inputDimensions
        let output = PTQ1MetalKernel.multiply(
            x.reshaped(rows, inputDimensions), weight: weight,
            inputDimensions: inputDimensions, outputDimensions: outputDimensions)
        return output.reshaped(Array(x.shape.dropLast()) + [outputDimensions]).asType(x.dtype)
    }
}

/// PTQ1 embedding lookup. The compact matrix is shared with `asLinear`, so a
/// tied vocabulary never materializes decoded weights.
open class PTQ1Embedding: Embedding {
    public let embeddingCount: Int
    public let dimensions: Int

    public override var shape: (Int, Int) { (embeddingCount, dimensions) }

    public override init(embeddingCount: Int, dimensions: Int) {
        precondition(dimensions > 0 && dimensions.isMultiple(of: PTQ1Linear.blockWidth))
        self.embeddingCount = embeddingCount
        self.dimensions = dimensions
        super.init(
            weight: MLXArray.zeros(
                [embeddingCount, dimensions / PTQ1Linear.blockWidth, PTQ1Linear.bytesPerBlock],
                dtype: .uint8))
    }

    public init(weight: MLXArray, embeddingCount: Int, dimensions: Int) {
        precondition(dimensions > 0 && dimensions.isMultiple(of: PTQ1Linear.blockWidth))
        precondition(
            weight.dtype == .uint8
                && weight.shape
                    == [
                        embeddingCount, dimensions / PTQ1Linear.blockWidth,
                        PTQ1Linear.bytesPerBlock,
                    ],
            "Invalid PTQ1_0 embedding tensor")
        self.embeddingCount = embeddingCount
        self.dimensions = dimensions
        super.init(weight: weight)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        PTQ1MetalKernel.lookup(
            x, weight: weight, dimensions: dimensions, embeddingCount: embeddingCount)
    }

    open override func asLinear(_ x: MLXArray) -> MLXArray {
        PTQ1MetalKernel.multiply(
            x.reshaped(-1, dimensions), weight: weight, inputDimensions: dimensions,
            outputDimensions: embeddingCount
        ).reshaped(Array(x.shape.dropLast()) + [embeddingCount]).asType(x.dtype)
    }
}

package enum PTQ1MetalKernel {
    private static let source = """
        const uint lane = thread_position_in_grid.x;
        const uint output_index = thread_position_in_grid.y;
        const uint row = thread_position_in_grid.z;
        const uint powers[5] = {1u, 3u, 9u, 27u, 81u};

        float accumulator = 0.0f;
        for (uint block_index = 0; block_index < Blocks; ++block_index) {
            const device uchar* block = packed
                + (output_index * Blocks + block_index) * 28u;
            const ushort scale_bits = ushort(block[26]) | (ushort(block[27]) << 8u);
            const float scale = float(as_type<half>(scale_bits));
            for (uint dimension = lane; dimension < 128u; dimension += 32u) {
                uint byte_index;
                uint trit;
                if (dimension < 80u) {
                    byte_index = dimension % 16u;
                    trit = dimension / 16u;
                } else if (dimension < 120u) {
                    const uint local = dimension - 80u;
                    byte_index = 16u + local % 8u;
                    trit = local / 8u;
                } else {
                    const uint local = dimension - 120u;
                    byte_index = 24u + local % 2u;
                    trit = local / 2u;
                }
                const uint remainder = (uint(block[byte_index]) * powers[trit]) & 255u;
                const int code = int((remainder * 3u) >> 8u) - 1;
                const uint input_index = block_index * 128u + dimension;
                accumulator += input[row * InputWidth + input_index] * float(code) * scale;
            }
        }
        accumulator = simd_sum(accumulator);
        if (thread_index_in_simdgroup == 0u) {
            output[row * OutputWidth + output_index] = accumulator;
        }
        """

    private static let lookupSource = """
        const uint dimension = thread_position_in_grid.x;
        const uint row = thread_position_in_grid.y;
        const uint token = uint(indices[row]);
        if (token >= EmbeddingCount || dimension >= Width) { return; }
        const uint block_index = dimension / 128u;
        const uint local = dimension % 128u;
        const device uchar* block = packed
            + (token * Blocks + block_index) * 28u;
        uint byte_index;
        uint trit;
        if (local < 80u) {
            byte_index = local % 16u;
            trit = local / 16u;
        } else if (local < 120u) {
            const uint tail = local - 80u;
            byte_index = 16u + tail % 8u;
            trit = tail / 8u;
        } else {
            const uint tail = local - 120u;
            byte_index = 24u + tail % 2u;
            trit = tail / 2u;
        }
        const uint powers[5] = {1u, 3u, 9u, 27u, 81u};
        const uint remainder = (uint(block[byte_index]) * powers[trit]) & 255u;
        const int code = int((remainder * 3u) >> 8u) - 1;
        const ushort scale_bits = ushort(block[26]) | (ushort(block[27]) << 8u);
        output[row * Width + dimension] = float(code) * float(as_type<half>(scale_bits));
        """

    nonisolated(unsafe) private static var kernels = [Int: MLXFast.MLXFastKernel]()
    nonisolated(unsafe) private static var lookupKernels = [Int: MLXFast.MLXFastKernel]()
    private static let lock = NSLock()

    private static func kernel(inputDimensions: Int) -> MLXFast.MLXFastKernel {
        lock.withLock {
            if let kernel = kernels[inputDimensions] { return kernel }
            let kernel = MLXFast.metalKernel(
                name: "ptq1_matmul_i\(inputDimensions)",
                inputNames: ["input", "packed"], outputNames: ["output"],
                source: source, ensureRowContiguous: true)
            kernels[inputDimensions] = kernel
            return kernel
        }
    }

    private static func lookupKernel(dimensions: Int) -> MLXFast.MLXFastKernel {
        lock.withLock {
            if let kernel = lookupKernels[dimensions] { return kernel }
            let kernel = MLXFast.metalKernel(
                name: "ptq1_embedding_d\(dimensions)", inputNames: ["indices", "packed"],
                outputNames: ["output"], source: lookupSource, ensureRowContiguous: true)
            lookupKernels[dimensions] = kernel
            return kernel
        }
    }

    static func multiply(
        _ input: MLXArray, weight: MLXArray, inputDimensions: Int, outputDimensions: Int
    ) -> MLXArray {
        precondition(input.ndim == 2 && input.dim(1) == inputDimensions)
        precondition(
            weight.dtype == .uint8
                && weight.shape
                    == [
                        outputDimensions, inputDimensions / PTQ1Linear.blockWidth,
                        PTQ1Linear.bytesPerBlock,
                    ]
        )
        return kernel(inputDimensions: inputDimensions)(
            [input.asType(.float32), weight],
            template: [
                ("Blocks", inputDimensions / PTQ1Linear.blockWidth),
                ("InputWidth", inputDimensions), ("OutputWidth", outputDimensions),
            ],
            grid: (32, outputDimensions, input.dim(0)), threadGroup: (32, 1, 1),
            outputShapes: [[input.dim(0), outputDimensions]], outputDTypes: [.float32]
        )[0]
    }

    static func lookup(
        _ indices: MLXArray, weight: MLXArray, dimensions: Int, embeddingCount: Int
    ) -> MLXArray {
        let flat = indices.flattened().asType(.int32)
        let output = lookupKernel(dimensions: dimensions)(
            [flat, weight],
            template: [
                ("Blocks", dimensions / PTQ1Linear.blockWidth), ("Width", dimensions),
                ("EmbeddingCount", embeddingCount),
            ],
            grid: (dimensions, flat.size, 1), threadGroup: (min(dimensions, 256), 1, 1),
            outputShapes: [[flat.size, dimensions]], outputDTypes: [.float32]
        )[0]
        return output.reshaped(indices.shape + [dimensions])
    }
}
