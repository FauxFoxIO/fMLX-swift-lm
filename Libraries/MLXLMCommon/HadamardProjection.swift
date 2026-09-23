// Copyright © 2026 Faux Fox.

import Foundation
import MLX
import MLXNN

/// A checkpoint-defined Walsh-Hadamard activation transform.
public struct HadamardTransformContract {
    public enum Direction: String, Sendable {
        case forward
        case inverse
    }

    public let identifier: String
    public let blockSize: Int
    public let signs: MLXArray
    public let direction: Direction

    public init(
        identifier: String, blockSize: Int, signs: MLXArray, direction: Direction
    ) {
        precondition(blockSize > 0 && blockSize.nonzeroBitCount == 1)
        precondition(signs.ndim == 1 && signs.dim(0).isMultiple(of: blockSize))
        self.identifier = identifier
        self.blockSize = blockSize
        self.signs = signs
        self.direction = direction
    }

    package func isCompatible(with other: Self) -> Bool {
        identifier == other.identifier && blockSize == other.blockSize
            && direction == other.direction && signs.shape == other.signs.shape
    }

    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        precondition(input.dim(-1) == signs.dim(0))
        let shape = input.shape
        let dtype = input.dtype
        var value = input.asType(.float32)
        let floatSigns = signs.asType(.float32)
        if direction == .forward {
            value = value * floatSigns
        }
        value = hadamardTransform(
            value.reshaped(-1, blockSize), scale: 1 / sqrt(Float(blockSize))
        ).reshaped(shape)
        if direction == .inverse {
            value = value * floatSigns
        }
        return value.asType(dtype)
    }
}

/// A quantized projection whose stored weights require a checkpoint-defined input transform.
open class HadamardQuantizedLinear: QuantizedLinear {
    public let transform: HadamardTransformContract

    public init(_ base: QuantizedLinear, transform: HadamardTransformContract) {
        self.transform = transform
        super.init(
            weight: base.weight, bias: base.bias, scales: base.scales, biases: base.biases,
            groupSize: base.groupSize, bits: base.bits, mode: base.mode,
            globalScale: base.globalScale)
        freeze()
    }

    package init(
        weight: MLXArray, bias: MLXArray?, scales: MLXArray, biases: MLXArray?,
        groupSize: Int, bits: Int, mode: QuantizationMode, globalScale: MLXArray?,
        transform: HadamardTransformContract
    ) {
        self.transform = transform
        super.init(
            weight: weight, bias: bias, scales: scales, biases: biases,
            groupSize: groupSize, bits: bits, mode: mode, globalScale: globalScale)
        freeze()
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        super.callAsFunction(transform(x))
    }
}

/// A quantized embedding stored in the transformed basis.
open class HadamardQuantizedEmbedding: QuantizedEmbedding {
    public let transform: HadamardTransformContract

    public init(_ base: QuantizedEmbedding, transform: HadamardTransformContract) {
        self.transform = transform
        super.init(
            weight: base.weight, scales: base.scales, biases: base.biases,
            groupSize: base.groupSize, bits: base.bits, mode: base.mode,
            globalScale: base.globalScale)
        freeze()
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        transform(super.callAsFunction(x))
    }

    open override func asLinear(_ x: MLXArray) -> MLXArray {
        let forward = HadamardTransformContract(
            identifier: transform.identifier, blockSize: transform.blockSize,
            signs: transform.signs, direction: .forward)
        return super.asLinear(forward(x))
    }
}

/// A compact PTQ1 projection stored in the checkpoint's transformed basis.
open class HadamardPTQ1Linear: PTQ1Linear {
    public let transform: HadamardTransformContract

    public init(
        inputDimensions: Int, outputDimensions: Int, transform: HadamardTransformContract
    ) {
        self.transform = transform
        super.init(inputDimensions: inputDimensions, outputDimensions: outputDimensions)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        super.callAsFunction(transform(x))
    }
}

/// Compact PTQ1 token embeddings with the checkpoint-defined inverse transform.
open class HadamardPTQ1Embedding: PTQ1Embedding {
    public let transform: HadamardTransformContract

    public init(
        embeddingCount: Int, dimensions: Int, transform: HadamardTransformContract
    ) {
        self.transform = transform
        super.init(embeddingCount: embeddingCount, dimensions: dimensions)
    }

    open override func callAsFunction(_ x: MLXArray) -> MLXArray {
        transform(super.callAsFunction(x))
    }

    open override func asLinear(_ x: MLXArray) -> MLXArray {
        let forward = HadamardTransformContract(
            identifier: transform.identifier, blockSize: transform.blockSize,
            signs: transform.signs, direction: .forward)
        return super.asLinear(forward(x))
    }
}
