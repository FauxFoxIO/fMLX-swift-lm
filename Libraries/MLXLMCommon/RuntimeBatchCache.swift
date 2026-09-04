// Copyright © 2026 Faux Fox.
// Per-slot batch-view design adapted from Osaurus AI's MIT-licensed BatchKVCache,
// vmlx-swift-lm 4546a5d720e7013adffdbddd728c6106e4f9e637. See runtime documentation.

import MLX

/// One batched projection forward, with cache-native attention for each independent row.
/// Keeping attention row-local supports different lengths and quantized KV without padding.
final class RuntimeBatchKVCache: BaseKVCache, BatchPositionedKVCache, KVCacheAttentionProtocol {
    let slots: [KVCache]
    var batchOffset: MLXArray { MLXArray(slots.map { Int32($0.offset) }) }
    override var ropeOffset: RoPEOffset { .batch(batchOffset) }

    init(_ slots: [KVCache]) {
        self.slots = slots
        super.init()
        offset = slots.map(\.offset).max() ?? 0
    }

    override func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        // updateAndAttend constructs each row's native mask at its own position.
        .none
    }

    func updateAndAttend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let output = slots.enumerated().map { row, cache in
            attentionWithCacheUpdate(
                queries: queries[row ..< row + 1], keys: keys[row ..< row + 1],
                values: values[row ..< row + 1], cache: cache, scale: scale,
                mask: cache.makeMask(n: queries.dim(2), windowSize: nil, returnArray: false))
        }
        return concatenated(output, axis: 0)
    }
}

/// Owns transient recurrent row packing; commits only after the whole forward settles.
struct RuntimeBatchCache {
    let rows: [[KVCache]]
    let cache: [KVCache]

    init(rows: [[KVCache]]) throws {
        guard let first = rows.first, rows.allSatisfy({ $0.count == first.count }) else {
            throw ConcurrentTextRuntimeError.unsupportedCache
        }
        self.rows = rows
        cache = try first.indices.map { layer in
            let leaves = rows.map { $0[layer] }
            if leaves.allSatisfy({ type(of: $0) == MambaCache.self }) {
                let merged = MambaCache()
                guard leaves.allSatisfy({ $0.state.count == 2 }) else {
                    throw ConcurrentTextRuntimeError.unsupportedCache
                }
                merged[0] = concatenated(leaves.map { $0.state[0] }, axis: 0)
                merged[1] = concatenated(leaves.map { $0.state[1] }, axis: 0)
                return merged
            }
            guard
                leaves.allSatisfy({
                    type(of: $0) == KVCacheSimple.self || type(of: $0) == QuantizedKVCache.self
                })
            else { throw ConcurrentTextRuntimeError.unsupportedCache }
            return RuntimeBatchKVCache(leaves)
        }
    }

    func commit() {
        for layer in cache.indices {
            guard let merged = cache[layer] as? MambaCache else { continue }
            for row in rows.indices {
                let target = rows[row][layer] as! MambaCache
                target[0] = merged[0]![row ..< row + 1]
                target[1] = merged[1]![row ..< row + 1]
                target.advance(1)
            }
        }
    }
}
