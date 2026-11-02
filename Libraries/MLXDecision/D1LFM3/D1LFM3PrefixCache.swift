// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXNN

/// What a `.causalState` pass over a state prefix leaves for later calls, so they compute only their
/// new tokens. Per layer: a full-attention layer's rotated keys and values for every cached position;
/// a sliding layer's for the last window − 1; a convolution's last taps − 1 input rows.
/// About 12 KB per cached token on d1-2B (the six full layers), so 8K tokens is about 100 MB.
public struct D1LFM3PrefixCache: @unchecked Sendable {
    /// Positions the cache covers. The next call's tokens start here.
    public let length: Int
    let keys: [Int: MLXArray]
    let values: [Int: MLXArray]
    let convolutions: [Int: MLXArray]

    /// Bytes of the cached arrays.
    public var byteCount: Int {
        (Array(keys.values) + Array(values.values) + Array(convolutions.values)).reduce(0) {
            $0 + $1.nbytes
        }
    }

    /// Writes the cache as safetensors. `metadata` should identify the model and the prefix, so a
    /// stale file is never reused.
    public func save(to url: URL, metadata: [String: String] = [:]) throws {
        var arrays = [String: MLXArray]()
        for (layer, array) in keys { arrays["\(layer).k"] = array }
        for (layer, array) in values { arrays["\(layer).v"] = array }
        for (layer, array) in convolutions { arrays["\(layer).c"] = array }
        var metadata = metadata
        metadata["length"] = String(length)
        try withError { try MLX.save(arrays: arrays, metadata: metadata, url: url) }
    }

    /// Reads a cache ``save(to:metadata:)`` wrote, with its metadata.
    public static func load(from url: URL) throws -> (D1LFM3PrefixCache, [String: String]) {
        let (arrays, metadata) = try withError { try loadArraysAndMetadata(url: url) }
        guard let length = metadata["length"].flatMap(Int.init) else {
            throw DecisionError.invalidInput("the prefix cache has no length")
        }
        var keys = [Int: MLXArray]()
        var values = [Int: MLXArray]()
        var convolutions = [Int: MLXArray]()
        for (name, array) in arrays {
            let parts = name.split(separator: ".")
            guard parts.count == 2, let layer = Int(parts[0]) else { continue }
            switch parts[1] {
            case "k": keys[layer] = array
            case "v": values[layer] = array
            case "c": convolutions[layer] = array
            default: continue
            }
        }
        eval(Array(keys.values) + Array(values.values) + Array(convolutions.values))
        return (
            D1LFM3PrefixCache(
                length: length, keys: keys, values: values, convolutions: convolutions),
            metadata
        )
    }
}

extension D1LFM3Text {
    /// A `.causalState` pass over `input` [1, N, D] at positions `cache.length ..< + N`. The first
    /// `stateLength` new positions are state; the rest are questions in `groups` [N]. Returns the hidden
    /// states at `slots` (indices into the new positions; nil skips the readout) and the cache extended
    /// by the new state positions.
    func extend(
        _ input: MLXArray, after cache: D1LFM3PrefixCache?, stateLength: Int, groups: MLXArray,
        slots: MLXArray?
    ) -> (hidden: MLXArray?, cache: D1LFM3PrefixCache) {
        let start = cache?.length ?? 0
        let count = input.dim(1)
        let stateEnd = start + stateLength
        let window = configuration.slidingWindow
        let fullFreqs = wavelengths("full_attention")
        let slidingFreqs = wavelengths("sliding_attention")

        let queries = MLXArray(Int32(start) ..< Int32(start + count))[0..., .newAxis]
        let queryGroups = groups[0..., .newAxis]
        func fullMask(keysFrom first: Int) -> MLXArray {
            let k = MLXArray(Int32(first) ..< Int32(start + count))[.newAxis]
            // Cached keys are state, which every query may read; their group never matters.
            let keyGroups =
                (first < start
                    ? concatenated([MLXArray.zeros([start - first], dtype: .int32) - 1, groups])
                    : groups)[.newAxis]
            return which(
                queries .< Int32(stateEnd), k .<= queries,
                (k .< Int32(stateEnd)) .|| (queryGroups .== keyGroups))
        }
        func slidingMask(keysFrom first: Int) -> MLXArray {
            let k = MLXArray(Int32(first) ..< Int32(start + count))[.newAxis]
            return (k .<= queries) .&& (k .> (queries - Int32(window)))
        }

        var keys = [Int: MLXArray]()
        var values = [Int: MLXArray]()
        var convolutions = [Int: MLXArray]()
        var h = input
        var hidden: MLXArray?
        for (index, layer) in layers.enumerated() {
            let x = layer.operatorNorm(h)
            let y: MLXArray
            switch layer.kind {
            case .conv:
                let (out, tail) = layer.conv!(
                    x, past: cache?.convolutions[index], through: stateLength)
                y = out
                convolutions[index] = tail
            case .full, .sliding:
                let attention = layer.attention!
                let sliding = layer.kind == .sliding
                let (q, k, v) = attention.project(
                    x, freqs: sliding ? slidingFreqs : fullFreqs, offset: start)
                let pastKeys = cache?.keys[index]
                let allKeys = pastKeys.map { concatenated([$0, k], axis: 2) } ?? k
                let allValues = cache?.values[index].map { concatenated([$0, v], axis: 2) } ?? v
                let first = start - (pastKeys?.dim(2) ?? 0)
                let mask = sliding ? slidingMask(keysFrom: first) : fullMask(keysFrom: first)
                y = attention.attend(
                    x, queries: q, keys: allKeys, values: allValues, mask: mask)
                let stateKeys = allKeys[0..., 0..., ..<(stateEnd - first)]
                let stateValues = allValues[0..., 0..., ..<(stateEnd - first)]
                keys[index] = sliding ? D1LFM3Text.last(window - 1, of: stateKeys) : stateKeys
                values[index] =
                    sliding ? D1LFM3Text.last(window - 1, of: stateValues) : stateValues
            }
            h = h + layer.postOperatorNorm(y)
            if index == layers.count - 1 {
                if let slots { hidden = embeddingNorm(layer.feed(h[0..., slots])[0]) }
                break
            }
            h = layer.feed(h)
            if evaluationInterval > 0, (index + 1) % evaluationInterval == 0 {
                eval(h)
            }
        }
        return (
            hidden,
            D1LFM3PrefixCache(
                length: stateEnd, keys: keys, values: values, convolutions: convolutions)
        )
    }

    /// The last `count` positions of keys or values [1, KV, n, hd].
    static func last(_ count: Int, of array: MLXArray) -> MLXArray {
        let total = array.dim(2)
        return total > count ? array[0..., 0..., (total - count)...] : array
    }
}

extension D1LFM3Engine {
    /// Computes a state prefix (all of `ids` is state) after `cache`, and returns the extended cache.
    /// `features` fill the image placeholders in `ids`.
    public func prefill(
        _ ids: [Int32], after cache: D1LFM3PrefixCache? = nil, features: MLXArray? = nil
    ) throws -> D1LFM3PrefixCache {
        try requireCausalState()
        guard !ids.isEmpty else {
            if let cache { return cache }
            throw DecisionError.invalidInput("a prefix needs at least one token")
        }
        let h = try embeddings(
            D1LFM3Sequence(ids: ids, stateLength: ids.count, groups: [], slots: []),
            features: features)
        let result = network.text.extend(
            h, after: cache, stateLength: ids.count,
            groups: MLXArray.zeros([ids.count], dtype: .int32), slots: nil)
        let cache = result.cache
        eval(Array(cache.keys.values) + Array(cache.values.values) + Array(cache.convolutions.values))
        return cache
    }

    /// The hidden states at `sequence`'s answer slots, [K, D] float32, computing only `sequence`'s
    /// tokens after `cache`. `sequence.stateLength` counts its own state tokens. Also returns the cache
    /// extended by that state, for the next call.
    public func hidden(
        _ sequence: D1LFM3Sequence, after cache: D1LFM3PrefixCache, features: MLXArray? = nil
    ) throws -> (hidden: MLXArray, cache: D1LFM3PrefixCache) {
        try requireCausalState()
        let length = sequence.ids.count
        guard length > 0, !sequence.slots.isEmpty,
            sequence.slots.allSatisfy({ $0 >= 0 && $0 < length }),
            sequence.stateLength <= length
        else {
            throw DecisionError.invalidInput("answer slots must fall inside the new tokens")
        }
        guard sequence.groups.isEmpty || sequence.groups.count == length else {
            throw DecisionError.invalidInput("groups must have one entry per position")
        }
        let groups =
            sequence.groups.isEmpty
            ? MLXArray.ones([length], dtype: .int32) : MLXArray(sequence.groups)
        let h = try embeddings(sequence, features: features)
        let result = network.text.extend(
            h, after: cache, stateLength: sequence.stateLength, groups: groups,
            slots: MLXArray(sequence.slots.map(Int32.init)))
        return (result.hidden!.asType(.float32), result.cache)
    }

    private func requireCausalState() throws {
        guard stateAttention == .causalState else {
            throw DecisionError.unsupportedInput(
                "prefix caching needs causal state attention; this model reads `set` attention")
        }
    }
}
