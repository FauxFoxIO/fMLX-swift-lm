// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXDecision
import MLXLMCommon
import MLXNN

/// `sensitivity --model DIR --data DIR --out FILE.json [--bits 4] [--group 64] [--imatrix FILE]
/// [--limit N]`: quantizes one tensor group at a time (the rest stay float) and measures how far the
/// option distributions move from the float model on the set.
func sensitivity(_ arguments: Arguments) throws {
    let engine = try D1LFM3Engine.load(
        directory: try arguments.url("model"), dtype: arguments.dtype,
        weightLoading: arguments.weightLoading)
    var set = try EvalSet(try arguments.url("data"))
    var items = set.items
    if let limit = arguments.int("limit") { items = Array(items.prefix(limit)) }
    if let step = arguments.int("stride") {
        items = stride(from: 0, to: items.count, by: step).map { items[$0] }
    }
    let entries = try items.map { try set.entry($0) }
    let setting = D1LFM3Configuration.Quantization.Setting(
        groupSize: arguments.int("group") ?? 64, bits: arguments.int("bits") ?? 4)
    let importance =
        try arguments.options["imatrix"].map {
            try loadArrays(url: URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath))
        } ?? [:]
    let temperature: Float = 0.717

    // Image features stay float: this measures the trunk.
    let features = try entries.map { entry -> MLXArray? in
        let f = try imageFeatures(engine, entry)
        if let f { eval(f) }
        return f
    }
    var hiddens = [MLXArray]()
    func logits() throws -> [[MLXArray]] {
        hiddens = []
        return try zip(entries, features).map { entry, f in
            let hidden = try engine.hidden(entry.sequence, features: f)
            let rows = entry.tokens.enumerated().map { k, tokens in
                engine.scores(hidden[k ..< k + 1], tokens: tokens)[0]
            }
            eval(rows, hidden)
            hiddens.append(hidden)
            return rows
        }
    }
    let reference = try logits()
    let referenceHidden = hiddens
    func distance(_ other: [[MLXArray]]) -> [String: Double] {
        var num = 0.0
        var den = 0.0
        for (a, b) in zip(referenceHidden, hiddens) {
            num += Double(((a - b) * (a - b)).sum().item(Float.self))
            den += Double((a * a).sum().item(Float.self))
        }
        var kl = 0.0
        var maxDelta: Float = 0
        var flips = 0
        var n = 0
        for (a, b) in zip(reference, other) {
            for (x, y) in zip(a, b) {
                let p = softmax(x / temperature)
                let lp = logSoftmax(x / temperature)
                let lq = logSoftmax(y / temperature)
                kl += Double((p * (lp - lq)).sum().item(Float.self))
                maxDelta = max(maxDelta, abs(x - y).max().item(Float.self))
                flips += argMax(x).item(Int.self) == argMax(y).item(Int.self) ? 0 : 1
                n += 1
            }
        }
        return [
            "kl": kl / Double(max(1, n)), "max_dlogit": Double(maxDelta), "flips": Double(flips),
            "hidden_rel_mse": num / max(den, 1e-30),
        ]
    }

    // Groups: per layer, the attention or conv input projections, its output projection, and the
    // feed-forward's gate/up and down projections; vision tower and projector as one group each.
    var groups = [(String, [String])]()
    let paths = engine.linearPaths.filter { $0.hasPrefix("language_model.") }
    var byLayer = [Int: [String]]()
    for path in paths {
        let parts = path.split(separator: ".")
        if parts.count > 2, let layer = Int(parts[2]) { byLayer[layer, default: []].append(path) }
    }
    for layer in byLayer.keys.sorted() {
        let ps = byLayer[layer]!
        func pick(_ suffixes: [String]) -> [String] {
            ps.filter { p in suffixes.contains { p.hasSuffix($0) } }
        }
        for (name, suffixes) in [
            ("qkv", ["q_proj", "k_proj", "gate_proj"]), ("v", ["v_proj"]), ("o", ["o_proj"]),
            ("conv_in", ["conv.in_proj"]), ("conv_out", ["conv.out_proj"]),
            ("ffn_up", ["w1", "w3"]), ("ffn_down", ["w2"]),
        ] {
            let selected = pick(suffixes)
            if !selected.isEmpty { groups.append(("L\(layer).\(name)", selected)) }
        }
    }
    if let only = arguments.options["only"] {
        groups = groups.filter { $0.0.wholeMatch(of: try! Regex(only)) != nil }
    }

    var results = [[String: Any]]()
    for (name, members) in groups {
        var replaced = [String: Linear]()
        var originals = [String: Linear]()
        var params = 0
        for path in members {
            guard let w = engine.linearWeight(path) else { continue }
            let q = D1LFM3Quantizer.quantize(
                w, setting: setting, method: .search, importance: importance[path],
                scaleType: engine.dtype)
            replaced[path] = QuantizedLinear(
                weight: q.weight, bias: nil, scales: q.scales, biases: q.biases,
                groupSize: setting.groupSize, bits: setting.bits)
            originals[path] = Linear(weight: w, bias: nil)
            params += w.size
        }
        engine.replaceLinear(replaced)
        var row = distance(try logits())
        engine.replaceLinear(originals)
        row["params"] = Double(params)
        var record: [String: Any] = ["group": name]
        for (k, v) in row { record[k] = v }
        results.append(record)
        log(
            "\(name): rel-mse \(row["hidden_rel_mse"]!) kl \(row["kl"]!) max|dl| \(row["max_dlogit"]!) flips \(Int(row["flips"]!))"
        )
    }

    if arguments.flags.contains("embedding") {
        let w = engine.embeddingWeight
        for bits in [4, 6, 8] {
            let s = D1LFM3Configuration.Quantization.Setting(
                groupSize: setting.groupSize, bits: bits)
            let q = D1LFM3Quantizer.quantize(
                w, setting: s, method: .search, scaleType: engine.dtype)
            engine.replaceEmbedding(
                QuantizedEmbedding(
                    weight: q.weight, scales: q.scales, biases: q.biases, groupSize: s.groupSize,
                    bits: bits))
            var row = distance(try logits())
            row["params"] = Double(w.size)
            var record: [String: Any] = ["group": "embedding.\(bits)bit"]
            for (k, v) in row { record[k] = v }
            results.append(record)
            log(
                "embedding \(bits)-bit: kl \(row["kl"]!) max|dl| \(row["max_dlogit"]!) flips \(Int(row["flips"]!))"
            )
        }
        engine.replaceEmbedding(Embedding(weight: w))
    }
    let data = try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted])
    try data.write(to: try arguments.url("out"))
}
