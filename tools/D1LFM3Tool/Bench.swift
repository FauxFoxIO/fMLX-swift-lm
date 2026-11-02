// Copyright © 2026 Faux Fox, Inc.

import Darwin
import Foundation
import MLX
import MLXDecision
import MLXLMCommon

func footprint() -> (current: Double, peak: Double) {
    var info = rusage_info_v6()
    let ok = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
            proc_pid_rusage(getpid(), RUSAGE_INFO_V6, $0)
        }
    }
    guard ok == 0 else { return (0, 0) }
    return (
        Double(info.ri_phys_footprint) / 1_048_576,
        Double(info.ri_lifetime_max_phys_footprint) / 1_048_576
    )
}

func median(_ values: [Double]) -> Double {
    let sorted = values.sorted()
    return sorted.isEmpty ? 0 : sorted[sorted.count / 2]
}

/// `bench --model DIR --data DIR [--runs N] [--dtype] [--cache-mb N] [--out FILE.json]`: cold load,
/// warm vision and trunk latency (natural length, padded to 1024 and 2048), and peak footprint, on
/// the first one-image sequence over 1,100 tokens (as `mem_probe.py` picks) and the first text one.
func bench(_ arguments: Arguments) throws {
    let base = footprint()
    let runs = arguments.int("runs") ?? 12
    var set = try EvalSet(try arguments.url("data"))
    var screenshot: EvalSet.Entry?
    var text: EvalSet.Entry?
    for item in set.items {
        let entry = try set.entry(item)
        if screenshot == nil, entry.grids.count == 1, entry.sequence.ids.count > 1100 {
            screenshot = entry
        }
        if text == nil, entry.pixels == nil, entry.sequence.ids.count <= 1024 { text = entry }
        if screenshot != nil, text != nil { break }
    }
    guard let screenshot, let text else { throw ToolError("the set has no suitable sequences") }
    let loadStart = DispatchTime.now().uptimeNanoseconds
    let engine = try D1LFM3Engine.load(
        directory: try arguments.url("model"), dtype: arguments.dtype,
        weightLoading: arguments.weightLoading)
    let loadMs = milliseconds(since: loadStart)
    engine.evaluationInterval = arguments.int("eval-every") ?? 0
    let loaded = footprint()

    func call(_ entry: EvalSet.Entry, pad: Int?) throws -> (vision: Double, trunk: Double) {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let features = try imageFeatures(engine, entry)
        if let features { eval(features) }
        let vision = milliseconds(since: t0)
        let t1 = DispatchTime.now().uptimeNanoseconds
        let hidden = try engine.hidden(entry.sequence, features: features, paddedLength: pad)
        let scores = engine.scores(hidden, tokens: entry.tokens[0])
        eval(scores)
        return (vision, milliseconds(since: t1))
    }

    let first = try call(screenshot, pad: 2048)
    let afterFirst = footprint()
    var results = [String: Any]()
    var vision = [Double]()
    for (name, entry, pad) in [
        ("trunk_l2048", screenshot, 2048), ("trunk_natural_screenshot", screenshot, nil),
        ("trunk_l1024", text, 1024), ("trunk_natural_text", text, nil),
    ] as [(String, EvalSet.Entry, Int?)] {
        var times = [Double]()
        for _ in 0 ..< runs {
            let t = try call(entry, pad: pad)
            times.append(t.trunk)
            if entry.pixels != nil { vision.append(t.vision) }
        }
        results["\(name)_ms_median"] = median(times)
        results["\(name)_ms_min"] = times.min() ?? 0
    }
    // `--pads 4096,8192`: the text call padded to each length, for the context-window sweep.
    for pad in (arguments.options["pads"] ?? "").split(separator: ",").compactMap({ Int($0) }) {
        var times = [Double]()
        for _ in 0 ..< max(3, runs / 2) {
            times.append(try call(text, pad: pad).trunk)
        }
        results["trunk_l\(pad)_ms_median"] = median(times)
        results["peak_after_l\(pad)_mb"] = footprint().peak
    }
    let end = footprint()
    results["warm_vision_ms_median"] = median(vision)
    results["load_ms"] = loadMs
    results["first_call_ms"] = first.vision + first.trunk
    results["baseline_mb"] = base.current
    results["after_load_mb"] = loaded.current
    results["after_first_call_mb"] = afterFirst.current
    results["end_mb"] = end.current
    results["peak_phys_footprint_mb"] = end.peak
    results["mlx_peak_mb"] = Double(Memory.peakMemory) / 1_048_576
    results["mlx_cache_limit_mb"] = Double(Memory.cacheLimit) / 1_048_576
    results["screenshot_tokens"] = screenshot.sequence.ids.count
    results["screenshot_grid"] = screenshot.grids[0]
    results["text_tokens"] = text.sequence.ids.count
    results["model"] = engine.directory.path
    results["mapped_mb"] = Double(engine.mappedBytes) / 1_048_576
    results["copied_tensors"] = engine.copiedTensors.count
    results["weight_loading"] = arguments.flags.contains("mapped") ? "memoryMapped" : "copy"
    results["dtype"] = String(describing: engine.dtype)
    results["evaluation_interval"] = engine.evaluationInterval

    // `footprint` splits the process's memory into file-backed and anonymous regions.
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/usr/bin/footprint")
    process.arguments = ["\(getpid())"]
    let pipe = Pipe()
    process.standardOutput = pipe
    process.standardError = pipe
    try process.run()
    let output = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
    process.waitUntilExit()
    results["footprint"] = output.split(separator: "\n").prefix(40).map(String.init)

    let data = try JSONSerialization.data(
        withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
    if let out = arguments.options["out"] {
        try data.write(to: URL(fileURLWithPath: (out as NSString).expandingTildeInPath))
    }
    print(String(decoding: data, as: UTF8.self))
}
