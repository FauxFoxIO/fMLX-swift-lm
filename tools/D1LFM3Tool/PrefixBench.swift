// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXDecision

/// `prefix-bench --model DIR [--mapped] [--dtype fp16] [--prefixes 2048,8192] [--suffixes 16,64,300,1000]
/// [--runs N] [--out FILE.json]`: checks that a cached call matches a full `.causalState` pass, then
/// times prefill, save and load of the cache, and calls that compute only their new tokens.
/// Synthetic token rows; the model reads them under causal state whatever its export says.
func prefixBench(_ arguments: Arguments) throws {
    let runs = arguments.int("runs") ?? 5
    let engine = try D1LFM3Engine.load(
        directory: try arguments.url("model"), dtype: arguments.dtype, vision: false,
        weightLoading: arguments.weightLoading)
    engine.evaluationInterval = arguments.int("eval-every") ?? 0
    engine.stateAttention = .causalState
    let list = { (name: String, fallback: [Int]) in
        arguments.options[name].map { $0.split(separator: ",").compactMap { Int($0) } } ?? fallback
    }
    let prefixes = list("prefixes", [2048, 4096, 8192, 16384])
    let suffixes = list("suffixes", [16, 64, 300, 1000])
    func ids(_ count: Int, from start: Int) -> [Int32] {
        (start ..< start + count).map { Int32(1000 + ($0 &* 7919) % 30000) }
    }
    /// `count` new tokens: all state but the last 16, which are two 8-token questions with a slot each.
    func suffix(_ count: Int, from start: Int) -> D1LFM3Sequence {
        let questions = min(16, count)
        let state = count - questions
        let groups = (0 ..< count).map { Int32($0 < state ? 0 : ($0 - state < questions / 2 ? 1 : 2)) }
        return D1LFM3Sequence(
            ids: ids(count, from: start), stateLength: state, groups: groups,
            slots: [state + questions / 2 - 1, count - 1])
    }
    func time(_ body: () throws -> Void) rethrows -> Double {
        let start = DispatchTime.now().uptimeNanoseconds
        try body()
        return milliseconds(since: start)
    }
    var results: [String: Any] = ["model": engine.directory.path, "dtype": String(describing: engine.dtype)]

    // Parity: full pass vs prefix (in two chunks) + cached call, and vs a saved and reloaded cache.
    do {
        let (p, n) = (1536, 300)
        let tail = suffix(n, from: p)
        let whole = D1LFM3Sequence(
            ids: ids(p, from: 0) + tail.ids, stateLength: p + tail.stateLength,
            groups: Array(repeating: 0, count: p) + tail.groups, slots: tail.slots.map { $0 + p })
        let reference = try engine.hidden(whole)
        let half = try engine.prefill(ids(p / 2, from: 0))
        let cache = try engine.prefill(ids(p - p / 2, from: p / 2), after: half)
        let cached = try engine.hidden(tail, after: cache).hidden
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("d1-prefix.safetensors")
        try cache.save(to: file, metadata: ["prefix": "parity"])
        let reloaded = try engine.hidden(tail, after: try D1LFM3PrefixCache.load(from: file).0).hidden
        try? FileManager.default.removeItem(at: file)
        func compare(_ a: MLXArray, _ b: MLXArray) -> [String: Double] {
            let difference = abs(a - b).max().item(Float.self)
            let cosine = (sum(a * b, axis: -1) / (sqrt(sum(a * a, axis: -1)) * sqrt(sum(b * b, axis: -1))))
                .min().item(Float.self)
            let scale = abs(a).max().item(Float.self)
            return ["max_abs": Double(difference), "max_ref": Double(scale), "min_cosine": Double(cosine)]
        }
        results["parity_cached_vs_full"] = compare(cached, reference)
        results["parity_reloaded_vs_cached"] = compare(reloaded, cached)
        log("parity cached=\(results["parity_cached_vs_full"]!) reloaded=\(results["parity_reloaded_vs_cached"]!)")
    }

    var rows = [[String: Any]]()
    for p in prefixes {
        var row: [String: Any] = ["prefix": p]
        var cache: D1LFM3PrefixCache?
        var prefillTimes = [Double]()
        for index in 0 ..< 3 {
            let t = try time { cache = try engine.prefill(ids(p, from: 0)) }
            if index > 0 { prefillTimes.append(t) }
        }
        let cacheReady = cache!
        row["prefill_ms"] = median(prefillTimes)
        row["cache_mb"] = Double(cacheReady.byteCount) / 1_048_576
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("d1-prefix-\(p).safetensors")
        row["save_ms"] = try time { try cacheReady.save(to: file) }
        row["file_mb"] = Double((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) / 1_048_576
        var loaded: D1LFM3PrefixCache?
        row["load_ms"] = try time { loaded = try D1LFM3PrefixCache.load(from: file).0 }
        try? FileManager.default.removeItem(at: file)
        var calls = [String: Double]()
        for n in suffixes {
            let sequence = suffix(n, from: p)
            var times = [Double]()
            for index in 0 ... runs {
                let t = try time { eval(try engine.hidden(sequence, after: loaded!).hidden) }
                if index > 0 { times.append(t) }
            }
            calls["\(n)"] = median(times)
        }
        row["call_ms"] = calls
        row["peak_mb"] = footprint().peak
        log("prefix=\(p) prefill=\(row["prefill_ms"]!)ms cache=\(row["cache_mb"]!)MB save=\(row["save_ms"]!) load=\(row["load_ms"]!) calls=\(calls)")
        rows.append(row)
    }
    results["prefixes"] = rows
    let data = try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
    if let out = arguments.options["out"] {
        try data.write(to: URL(fileURLWithPath: (out as NSString).expandingTildeInPath))
    }
    print(String(decoding: data, as: UTF8.self))
}
