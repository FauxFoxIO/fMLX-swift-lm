// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXNN

/// A model that can split its weights across a tensor-parallel group.
///
/// Every rank loads the same checkpoint, then ``shard(group:)`` replaces each rank's
/// parameters with its slice. The loader calls it after the checkpoint update and before the
/// weights are evaluated, so a rank only materializes its own slice. Set
/// ``ModelConfiguration/tensorParallelGroup`` to load a sharded model through a factory.
///
/// The usual Megatron layout applies:
///
/// - attention: `q`/`k`/`v` are `allToSharded` (each rank owns whole heads),
///   `o` is `shardedToAll`
/// - dense MLP: `gate`/`up` are all-to-sharded, `down` is sharded-to-all
/// - MoE: every expert is split the same way with ``SwitchGLU/shard(group:)``, and the
///   MoE block sums its output across the group once
/// - embeddings, norms, routers and the LM head stay whole on every rank
///
/// Every rank must call ``shard(group:)`` with the same group, and every forward must run on
/// every rank in the same order, since the sharded layers communicate.
public protocol ShardableModel: AnyObject {
    /// The group this model was sharded across, or `nil` before ``shard(group:)``.
    var tensorParallelGroup: MLXDistributed.Group? { get }

    /// Replace this rank's parameters with its slice of the weights.
    func shard(group: MLXDistributed.Group) throws
}

public enum TensorParallelError: LocalizedError, Equatable {
    /// A tensor-parallel group was requested for a model without ``ShardableModel``.
    case modelNotShardable(String)
    /// Attention heads must divide evenly so that each rank owns whole heads.
    case indivisibleHeads(heads: Int, kvHeads: Int, groupSize: Int)
    /// A dimension cannot be split across the group.
    case indivisible(name: String, dimension: Int, groupSize: Int)
    /// A module that ``shardLinears(in:group:_:)`` should replace is missing or is not a `Linear`.
    case missingLinear(String)
    /// A request could not be exchanged between ranks.
    case invalidMessage(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotShardable(let type):
            "\(type) does not support tensor parallelism (it does not conform to ShardableModel)"
        case .indivisibleHeads(let heads, let kvHeads, let groupSize):
            "cannot shard \(heads) attention heads and \(kvHeads) KV heads across \(groupSize) ranks: "
                + "both must be divisible by the group size"
        case .indivisible(let name, let dimension, let groupSize):
            "cannot shard \(name) of size \(dimension) across \(groupSize) ranks"
        case .missingLinear(let key):
            "no Linear child named '\(key)' to shard"
        case .invalidMessage(let message):
            "invalid tensor-parallel message: \(message)"
        }
    }
}

/// Check that attention heads split evenly across `group`.
///
/// A projection can divide evenly while its heads do not, for example one KV head of 128
/// dimensions across 2 ranks. That would cut a head in half, so check heads directly.
public func validateHeadSharding(heads: Int, kvHeads: Int, group: MLXDistributed.Group) throws {
    let size = group.size
    guard heads % size == 0, kvHeads % size == 0 else {
        throw TensorParallelError.indivisibleHeads(heads: heads, kvHeads: kvHeads, groupSize: size)
    }
}

/// Replace the `Linear` children of `module` named in `plan` with sharded layers.
///
/// Quantized layers become the quantized sharded flavor, so their scales and biases are sliced
/// with the packed weight. Keys are paths relative to `module`, such as `"q_proj"` or
/// `"mlp.down_proj"`. Every key must name a `Linear`.
public func shardLinears(
    in module: Module, group: MLXDistributed.Group, _ plan: [String: ShardingType]
) throws {
    let leaves = Dictionary(
        module.leafModules().flattened(), uniquingKeysWith: { first, _ in first })
    var updates = [(String, Module)]()
    // sorted so that every rank builds its layers in the same order
    for (key, sharding) in plan.sorted(by: { $0.key < $1.key }) {
        guard let linear = leaves[key] as? Linear else {
            throw TensorParallelError.missingLinear(key)
        }
        updates.append((key, try shardLinear(linear, sharding: sharding, group: group)))
    }
    try module.update(modules: ModuleChildren.unflattened(updates), verify: .none)
}

extension MLXDistributed {
    /// Every rank receives rank `root`'s value of `x`.
    ///
    /// This is a masked `allSum`: the other ranks contribute zeros. It is
    /// lazy like every collective, so it can sit inside an `asyncEval` pipeline.
    public static func broadcast(
        _ x: MLXArray, from root: Int = 0, group: Group, stream: StreamOrDevice = .cpu
    ) -> MLXArray {
        guard group.size > 1 else { return x }
        // Keep `x` as an input on every rank. Plain zeros would let a rank run this
        // collective before the ones `x` depends on, and the ranks would deadlock.
        let contribution = MLX.where(MLXArray(group.rank == root), x, MLXArray.zeros(like: x))
        return allSum(contribution, group: group, stream: stream)
    }

    /// Block until every rank in `group` reaches this point.
    public static func barrier(group: Group) {
        guard group.size > 1 else { return }
        eval(allSum(MLXArray(Int32(1)), group: group, stream: .cpu))
    }
}

/// The group a model's sampled tokens must agree across, or `nil` for a single rank.
func tensorParallelTokenGroup(of model: any LanguageModel) -> MLXDistributed.Group? {
    guard let group = (model as? any ShardableModel)?.tensorParallelGroup, group.size > 1 else {
        return nil
    }
    return group
}

/// Evaluate `arrays` in groups of about `maxBytes`, so that the lazily loaded full tensors
/// behind sharded slices are freed as each group completes.
func evaluateInBatches(_ arrays: [MLXArray], maxBytes: Int = 1 << 30) {
    var batch = [MLXArray]()
    var bytes = 0
    for array in arrays {
        batch.append(array)
        bytes += array.nbytes
        if bytes >= maxBytes {
            eval(batch)
            batch.removeAll(keepingCapacity: true)
            bytes = 0
        }
    }
    if !batch.isEmpty {
        eval(batch)
    }
}

// MARK: - Mixture of experts

extension SwitchLinear {
    /// Shard every expert of this layer in place.
    ///
    /// The layer does not communicate. The MoE block that owns it sums its output across the
    /// group once, after the routed experts (and any shared expert) are combined.
    public func shard(_ sharding: ShardingType, group: MLXDistributed.Group) throws {
        let size = group.size
        guard size > 1 else { return }

        switch sharding {
        case .allToSharded:
            guard outputDims % size == 0 else {
                throw TensorParallelError.indivisible(
                    name: "expert output", dimension: outputDims, groupSize: size)
            }
        case .shardedToAll:
            guard inputDims % size == 0 else {
                throw TensorParallelError.indivisible(
                    name: "expert input", dimension: inputDims, groupSize: size)
            }
            // a quantized slice must hold whole quantization groups
            if let quantized = self as? QuantizedSwitchLinear,
                quantized.scales.dim(-1) % size != 0
            {
                throw TensorParallelError.indivisible(
                    name: "expert quantization groups", dimension: quantized.scales.dim(-1),
                    groupSize: size)
            }
        }

        try shardInPlace(self, sharding: sharding, group: group)

        switch sharding {
        case .allToSharded:
            outputDims /= size
        case .shardedToAll:
            inputDims /= size
            // the bias belongs to the summed output, so only one rank adds it
            if let bias, group.rank != 0 {
                update(parameters: ModuleParameters(values: ["bias": .value(zeros(like: bias))]))
            }
        }
    }
}

extension SwitchGLU {
    /// Split every expert across `group`: `gate`/`up` by output, `down` by input.
    ///
    /// The result on each rank is a partial sum. The owning MoE block must apply
    /// `MLXDistributed.allSum` to its combined output.
    public func shard(group: MLXDistributed.Group) throws {
        try gateProj.shard(.allToSharded, group: group)
        try upProj.shard(.allToSharded, group: group)
        try downProj.shard(.shardedToAll, group: group)
    }
}

extension FusedGateUpSwitchGLU {
    /// Split every expert across `group`, keeping the gate and up halves of `gate_up_proj`
    /// paired on each rank. The owning MoE block must sum its output across the group.
    public func shard(group: MLXDistributed.Group) throws {
        let size = group.size
        guard size > 1 else { return }
        guard hiddenDims % size == 0 else {
            throw TensorParallelError.indivisible(
                name: "expert hidden", dimension: hiddenDims, groupSize: size)
        }
        try shardInPlace(gateUpProj, sharding: .allToSharded, segments: .count(2), group: group)
        gateUpProj.outputDims /= size
        try downProj.shard(.shardedToAll, group: group)
    }
}
