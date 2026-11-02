// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXDecision
import MLXLMCommon

/// `convert --checkpoint DIR --out DIR [--dtype bf16|fp16] [--recipe FILE.json] [--imatrix FILE]
/// [--text-only]`
func convert(_ arguments: Arguments) throws {
    let recipe = try arguments.options["recipe"].map {
        try JSONDecoder().decode(
            D1LFM3Recipe.self,
            from: Data(contentsOf: URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath)))
    }
    let importance =
        try arguments.options["imatrix"].map {
            try loadArrays(url: URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath))
        } ?? [:]
    let start = DispatchTime.now().uptimeNanoseconds
    let report = try D1LFM3Export.convert(
        checkpoint: try arguments.url("checkpoint"), output: try arguments.url("out"),
        dtype: arguments.dtype ?? .bfloat16, recipe: recipe, importance: importance,
        includeVision: !arguments.flags.contains("text-only"))
    var counts = [String: Int]()
    for kind in report.tensors.values { counts[kind, default: 0] += 1 }
    log(
        "wrote \(report.bytes / 1_048_576) MB in \(Int(milliseconds(since: start) / 1000)) s: \(counts)"
    )
    if arguments.flags.contains("verbose") {
        for (path, kind) in report.tensors.sorted(by: { $0.key < $1.key }) {
            log("  \(path): \(kind)")
        }
    }
}

/// `imatrix --model DIR --data DIR --out FILE [--limit N]`: mean squared input channel of every
/// linear layer over the set's sequences.
func collectImportance(_ arguments: Arguments) throws {
    let engine = try D1LFM3Engine.load(
        directory: try arguments.url("model"), dtype: arguments.dtype,
        weightLoading: arguments.weightLoading)
    var set = try EvalSet(try arguments.url("data"))
    var items = set.items
    if let limit = arguments.int("limit") { items = Array(items.prefix(limit)) }
    var tokens = 0
    let importance = try engine.collectImportance {
        for (i, item) in items.enumerated() {
            let entry = try set.entry(item)
            let features = try imageFeatures(engine, entry)
            if let features { eval(features) }
            eval(try engine.hidden(entry.sequence, features: features))
            tokens += entry.sequence.ids.count
            if (i + 1) % 25 == 0 { log("\(i + 1)/\(items.count)") }
        }
    }
    try save(arrays: importance, url: try arguments.url("out"))
    log("wrote importance for \(importance.count) layers over \(tokens) tokens")
}
