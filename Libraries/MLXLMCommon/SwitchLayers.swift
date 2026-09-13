import Foundation
import MLX
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

public let compiledSiluProduct: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { gate, up in
    MLXNN.silu(gate) * up
}

private let weightedMultipleExpertSum: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { outputs, weights in
    (outputs * MLX.expandedDimensions(weights, axis: -1)).sum(axis: -2)
}

private let weightedSingleExpertSum: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { outputs, weights in
    MLX.squeezed(outputs * MLX.expandedDimensions(weights, axis: -1), axis: -2)
}

public let weightedExpertSum: @Sendable (MLXArray, MLXArray) -> MLXArray = { outputs, weights in
    if outputs.dim(-2) == 1 {
        weightedSingleExpertSum(outputs, weights)
    } else {
        weightedMultipleExpertSum(outputs, weights)
    }
}

public func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()
    let order = argSort(indices)
    let inverseOrder = argSort(order)

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}

/// Fused inverse-permutation and weighted reduction for sorted MoE rows.
///
/// `SwitchGLU` sorts expert assignments before its gathered matrix
/// multiplications. The established path restores a full
/// `[tokens, topK, hidden]` tensor and then reduces `topK`. This kernel reads
/// the sorted rows through the inverse permutation and writes
/// `[tokens, hidden]` directly, avoiding that intermediate allocation.
private let weightedExpertUnsortKernel = MLXFast.metalKernel(
    name: "weighted_expert_unsort",
    inputNames: ["sorted_outputs", "inverse_order", "weights"],
    outputNames: ["output"],
    source: """
            uint feature = thread_position_in_grid.x;
            uint token = thread_position_in_grid.y;

            T accumulator = (T)0;
            const uint assignment_base = token * (uint)K;
            for (uint slot = 0; slot < (uint)K; ++slot) {
                const uint assignment = assignment_base + slot;
                const uint sorted_row = (uint)inverse_order[assignment];
                // Match the legacy bfloat16 multiply-then-reduce rounding.
                const T weighted = (T)(
                    (float)sorted_outputs[sorted_row * threads_per_grid.x + feature]
                    * (float)weights[assignment]);
                accumulator = accumulator + weighted;
            }
            output[token * threads_per_grid.x + feature] = accumulator;
        """,
    ensureRowContiguous: true)

/// Reduce sorted top-8 bfloat16 expert rows without materializing their
/// unsorted assignment tensor. Callers must retain the established path for
/// every unsupported dtype, shape, or training topology.
package func weightedExpertUnsort(
    sortedOutputs: MLXArray,
    inverseOrder: MLXArray,
    weights: MLXArray
) -> MLXArray {
    let hidden = sortedOutputs.dim(1)
    precondition(
        sortedOutputs.ndim == 2 && hidden.isMultiple(of: 64)
            && sortedOutputs.dtype == .bfloat16,
        "weightedExpertUnsort requires bfloat16 [assignments, hidden], hidden % 64 == 0")
    precondition(
        inverseOrder.ndim == 1 && inverseOrder.dtype == .uint32,
        "weightedExpertUnsort requires flat uint32 inverse order")
    precondition(
        weights.ndim == 2 && weights.dim(1) == 8 && weights.size >= 64
            && weights.dtype == .bfloat16,
        "weightedExpertUnsort requires sorted-prefill bfloat16 [tokens, 8]")
    precondition(
        sortedOutputs.dim(0) == weights.size && inverseOrder.size == weights.size,
        "weightedExpertUnsort assignment counts must match")

    let tokens = weights.dim(0)
    return weightedExpertUnsortKernel(
        [sortedOutputs, inverseOrder, weights],
        template: [("T", sortedOutputs.dtype), ("K", 8)],
        grid: (hidden, tokens, 1),
        threadGroup: (64, 4, 1),
        outputShapes: [[tokens, hidden]],
        outputDTypes: [.bfloat16]
    )[0]
}

// MARK: - SwitchGLU

open class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = MLXNN.silu
        self.activationProduct = compiledSiluProduct

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.activationProduct = nil

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    /// Transform the expanded input ahead of the expert gather/sort.
    ///
    /// This pair of hooks exists so subclasses can wrap the expert
    /// projections without copying `projectExperts`' dataflow — and
    /// silently detaching from future changes to it, e.g. the gather/sort
    /// threshold or the compiled activation product. `RotateSwitchGLU`
    /// rotates activations here; the identity defaults add no graph nodes.
    func transformInput(_ x: MLXArray) -> MLXArray { x }

    /// Transform the activated hidden state ahead of `downProj`.
    /// Identity by default — see `transformInput`.
    func transformHidden(_ x: MLXArray) -> MLXArray { x }

    private func projectExperts(
        _ x: MLXArray, _ indices: MLXArray
    ) -> (output: MLXArray, inverseOrder: MLXArray?) {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])
        x = transformInput(x)

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let xUp = upProj(x, idx, sortedIndices: doSort)
        let xGate = gateProj(x, idx, sortedIndices: doSort)
        var activated =
            if let activationProduct {
                activationProduct(xGate, xUp)
            } else {
                activation(xGate) * xUp
            }
        activated = transformHidden(activated)
        x = downProj(
            activated,
            idx,
            sortedIndices: doSort)

        return (x, doSort ? inverseOrder : nil)
    }

    private func legacyWeightedReduction(
        _ projected: (output: MLXArray, inverseOrder: MLXArray?),
        indices: MLXArray,
        weights: MLXArray
    ) -> MLXArray {
        var output = projected.output
        if let inverseOrder = projected.inverseOrder {
            output = scatterUnsort(x: output, invOrder: inverseOrder, shape: indices.shape)
        }
        return weightedExpertSum(MLX.squeezed(output, axis: -2), weights)
    }

    /// Whether this call has the exact frozen, quantized inference topology
    /// supported by ``weightedExpertUnsort``.
    package func supportsDirectWeightedReduction(
        _ x: MLXArray, _ indices: MLXArray, weights: MLXArray
    ) -> Bool {
        let projections = [gateProj, upProj, downProj]
        return inputDims.isMultiple(of: 64)
            && x.ndim == 2
            && x.dim(1) == inputDims
            && x.dtype == .bfloat16
            && indices.ndim == 2
            && indices.dim(0) == x.dim(0)
            && indices.dim(1) == 8
            && indices.dtype == .uint32
            && weights.shape == indices.shape
            && weights.dtype == .bfloat16
            && indices.size >= 64
            && projections.allSatisfy {
                ObjectIdentifier(type(of: $0)) == ObjectIdentifier(QuantizedSwitchLinear.self)
                    && $0.bias == nil
            }
            && trainableParameters().flattened().isEmpty
    }

    open func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var projected = projectExperts(x, indices)

        if let inverseOrder = projected.inverseOrder {
            projected.output = scatterUnsort(
                x: projected.output, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(projected.output, axis: -2)
    }

    /// Project and combine selected experts, directly reducing sorted
    /// production prefill rows when requested and eligible.
    ///
    /// Disabled, decode-sized, non-bfloat16, custom, and trainable calls use
    /// the established scatter + ``weightedExpertSum`` path unchanged.
    package func callAndWeightedReduce(
        _ x: MLXArray,
        _ indices: MLXArray,
        weights: MLXArray,
        fuseSortedReduction: Bool
    ) -> MLXArray {
        guard fuseSortedReduction,
            supportsDirectWeightedReduction(x, indices, weights: weights)
        else {
            return weightedExpertSum(callAsFunction(x, indices), weights)
        }

        let projected = projectExperts(x, indices)
        guard let inverseOrder = projected.inverseOrder,
            projected.output.ndim == 3,
            projected.output.dim(-2) == 1,
            projected.output.dim(-1) == inputDims,
            projected.output.dtype == .bfloat16
        else {
            return legacyWeightedReduction(projected, indices: indices, weights: weights)
        }

        return weightedExpertUnsort(
            sortedOutputs: MLX.squeezed(projected.output, axis: -2),
            inverseOrder: inverseOrder,
            weights: weights)
    }
}

// MARK: - FusedGateUpSwitchGLU

/// SwitchGLU variant for models that ship a single fused `gate_up_proj` weight
/// of shape `[numExperts, 2*hiddenDims, inputDims]` instead of separate
/// `gate_proj` / `up_proj`. Used by Gemma 4 26B MoE.
open class FusedGateUpSwitchGLU: Module {
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = MLXNN.silu
        self.activationProduct = compiledSiluProduct

        self._gateUpProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: 2 * hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.activationProduct = nil

        self._gateUpProj.wrappedValue = SwitchLinear(
            inputDims: inputDims, outputDims: 2 * hiddenDims, numExperts: numExperts, bias: bias)
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    open func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let gateUp = gateUpProj(x, idx, sortedIndices: doSort)
        let parts = MLX.split(gateUp, parts: 2, axis: -1)
        let activated =
            if let activationProduct {
                activationProduct(parts[0], parts[1])
            } else {
                activation(parts[0]) * parts[1]
            }
        x = downProj(
            activated,
            idx,
            sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

// MARK: - SwitchLinear

open class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    open func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

open class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    override open func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        var result = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }
}

// MARK: - Streamed SwitchGLU

/// One quantized routed expert materialized from its independent weight ranges.
///
/// Each array excludes the expert axis. ``StreamedSwitchGLU`` stacks only the
/// slots selected for one small decode call, then remaps the router IDs to that
/// compact axis before using the same quantized gather kernels as ``SwitchGLU``.
/// Scales and affine biases may be FP16 or BF16; biases are optional.
public struct StreamedQuantizedExpertWeights: @unchecked Sendable {
    public let gateWeight: MLXArray
    public let gateScales: MLXArray
    public let gateBiases: MLXArray?
    public let upWeight: MLXArray
    public let upScales: MLXArray
    public let upBiases: MLXArray?
    public let downWeight: MLXArray
    public let downScales: MLXArray
    public let downBiases: MLXArray?
    public let byteCount: Int

    public init(
        gateWeight: MLXArray,
        gateScales: MLXArray,
        gateBiases: MLXArray? = nil,
        upWeight: MLXArray,
        upScales: MLXArray,
        upBiases: MLXArray? = nil,
        downWeight: MLXArray,
        downScales: MLXArray,
        downBiases: MLXArray? = nil,
        byteCount: Int
    ) throws {
        guard byteCount > 0 else { throw StreamedSwitchGLUError.invalidExpertByteCount(byteCount) }
        self.gateWeight = gateWeight
        self.gateScales = gateScales
        self.gateBiases = gateBiases
        self.upWeight = upWeight
        self.upScales = upScales
        self.upBiases = upBiases
        self.downWeight = downWeight
        self.downScales = downScales
        self.downBiases = downBiases
        self.byteCount = byteCount
    }
}

/// Validation failures for a streamed compact SwitchGLU invocation.
public enum StreamedSwitchGLUError: Error, Equatable, Sendable {
    case invalidExpertByteCount(Int)
    case unsupportedSequenceLength(Int)
    case invalidInputShape
    case invalidTopK(Int)
    case invalidRouterAssignmentCount(expected: Int, actual: Int)
    case invalidCompactRouterID(UInt32)
    case invalidCompactRouterMapping
    case invalidScoreShape
    case invalidExpertShape
}

/// Exact affine-quantized SwitchGLU execution over a compact expert axis.
///
/// This is intentionally not compiled. Its expert arrays are created after
/// range reads and must remain explicit inputs to the MLX graph; a compiled
/// body would otherwise capture the first materialized expert set as constants.
/// It accepts only one- and two-row decode calls. Prefill must retain the
/// resident expert layer rather than changing the width of its GEMM kernels.
public struct StreamedSwitchGLU: @unchecked Sendable {
    public let inputDims: Int
    public let hiddenDims: Int
    public let groupSize: Int
    public let bits: Int

    public init(inputDims: Int, hiddenDims: Int, groupSize: Int = 32, bits: Int = 4) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.groupSize = groupSize
        self.bits = bits
    }

    /// Runs the existing quantized gather path with a first-occurrence compact
    /// expert list and row-major remapped router IDs.
    public func callAndWeightedReduce(
        _ x: MLXArray,
        compactRouterIDs: [UInt32],
        topK: Int,
        weights: MLXArray,
        experts: [StreamedQuantizedExpertWeights]
    ) throws -> MLXArray {
        guard x.ndim == 2 else { throw StreamedSwitchGLUError.invalidInputShape }
        let tokenCount = x.dim(0)
        guard (1 ... 2).contains(tokenCount) else {
            throw StreamedSwitchGLUError.unsupportedSequenceLength(tokenCount)
        }
        guard x.dim(1) == inputDims else {
            throw StreamedSwitchGLUError.invalidInputShape
        }
        guard topK > 0 else { throw StreamedSwitchGLUError.invalidTopK(topK) }
        let (expectedAssignmentCount, overflow) = tokenCount.multipliedReportingOverflow(by: topK)
        guard !overflow, compactRouterIDs.count == expectedAssignmentCount else {
            throw StreamedSwitchGLUError.invalidRouterAssignmentCount(
                expected: overflow ? .max : expectedAssignmentCount,
                actual: compactRouterIDs.count)
        }
        guard !experts.isEmpty else {
            throw StreamedSwitchGLUError.invalidRouterAssignmentCount(
                expected: expectedAssignmentCount, actual: 0)
        }
        guard compactRouterIDs.allSatisfy({ Int($0) < experts.count }) else {
            throw StreamedSwitchGLUError.invalidCompactRouterID(
                compactRouterIDs.first { Int($0) >= experts.count }!)
        }
        var seenSlots = Set<UInt32>()
        for slot in compactRouterIDs where seenSlots.insert(slot).inserted {
            guard slot == UInt32(seenSlots.count - 1) else {
                throw StreamedSwitchGLUError.invalidCompactRouterMapping
            }
        }
        guard seenSlots.count == experts.count else {
            throw StreamedSwitchGLUError.invalidCompactRouterMapping
        }
        guard weights.ndim == 2, weights.dim(0) == tokenCount, weights.dim(1) == topK else {
            throw StreamedSwitchGLUError.invalidScoreShape
        }
        guard experts.allSatisfy(isValid) else {
            throw StreamedSwitchGLUError.invalidExpertShape
        }

        let indices = MLXArray(compactRouterIDs).reshaped(tokenCount, topK)
        var expanded = MLX.expandedDimensions(x, axes: [-2, -3])
        let gateWeight = MLX.stacked(experts.map(\.gateWeight))
        let gateScales = MLX.stacked(experts.map(\.gateScales))
        let gateBiases = try stackOptional(experts.map(\.gateBiases))
        let upWeight = MLX.stacked(experts.map(\.upWeight))
        let upScales = MLX.stacked(experts.map(\.upScales))
        let upBiases = try stackOptional(experts.map(\.upBiases))
        let downWeight = MLX.stacked(experts.map(\.downWeight))
        let downScales = MLX.stacked(experts.map(\.downScales))
        let downBiases = try stackOptional(experts.map(\.downBiases))

        let up = MLX.gatherQuantizedMM(
            expanded, upWeight, scales: upScales, biases: upBiases, rhsIndices: indices,
            transpose: true, groupSize: groupSize, bits: bits, mode: .affine, sortedIndices: false)
        let gate = MLX.gatherQuantizedMM(
            expanded, gateWeight, scales: gateScales, biases: gateBiases, rhsIndices: indices,
            transpose: true, groupSize: groupSize, bits: bits, mode: .affine, sortedIndices: false)
        expanded = MLX.gatherQuantizedMM(
            compiledSiluProduct(gate, up), downWeight, scales: downScales, biases: downBiases,
            rhsIndices: indices, transpose: true, groupSize: groupSize, bits: bits, mode: .affine,
            sortedIndices: false)

        return weightedExpertSum(MLX.squeezed(expanded, axis: -2), weights)
    }

    private func isValid(_ expert: StreamedQuantizedExpertWeights) -> Bool {
        guard inputDims > 0, hiddenDims > 0, groupSize > 0 else { return false }
        let packedInputDims = inputDims / 8
        let packedHiddenDims = hiddenDims / 8
        let inputGroups = inputDims / groupSize
        let hiddenGroups = hiddenDims / groupSize
        return inputDims.isMultiple(of: 8)
            && hiddenDims.isMultiple(of: 8)
            && inputDims.isMultiple(of: groupSize)
            && hiddenDims.isMultiple(of: groupSize)
            && bits == 4
            && expert.gateWeight.dtype == .uint32
            && expert.gateWeight.shape == [hiddenDims, packedInputDims]
            && validQuantizationParameter(
                expert.gateScales, expectedShape: [hiddenDims, inputGroups])
            && validOptionalBias(
                expert.gateBiases, scales: expert.gateScales,
                expectedShape: [hiddenDims, inputGroups])
            && expert.upWeight.dtype == .uint32
            && expert.upWeight.shape == [hiddenDims, packedInputDims]
            && validQuantizationParameter(
                expert.upScales, expectedShape: [hiddenDims, inputGroups])
            && validOptionalBias(
                expert.upBiases, scales: expert.upScales,
                expectedShape: [hiddenDims, inputGroups])
            && expert.downWeight.dtype == .uint32
            && expert.downWeight.shape == [inputDims, packedHiddenDims]
            && validQuantizationParameter(
                expert.downScales, expectedShape: [inputDims, hiddenGroups])
            && validOptionalBias(
                expert.downBiases, scales: expert.downScales,
                expectedShape: [inputDims, hiddenGroups])
    }

    private func validQuantizationParameter(_ value: MLXArray, expectedShape: [Int]) -> Bool {
        (value.dtype == .float16 || value.dtype == .bfloat16) && value.shape == expectedShape
    }

    private func validOptionalBias(
        _ bias: MLXArray?, scales: MLXArray, expectedShape: [Int]
    ) -> Bool {
        guard let bias else { return true }
        return bias.dtype == scales.dtype && bias.shape == expectedShape
    }

    private func stackOptional(_ arrays: [MLXArray?]) throws -> MLXArray? {
        let present = arrays.compactMap { $0 }
        guard present.isEmpty || present.count == arrays.count else {
            throw StreamedSwitchGLUError.invalidExpertShape
        }
        return present.isEmpty ? nil : MLX.stacked(present)
    }
}
