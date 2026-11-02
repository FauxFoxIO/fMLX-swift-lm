// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXDecision

/// Sequences exported by `export_sets.py`: safetensors shards plus `index.json`.
struct EvalSet {
    struct Item: Decodable {
        let key: Int
        let shard: String
        let slots: Int
        let kind: String
    }

    struct Entry {
        let item: Item
        let sequence: D1LFM3Sequence
        let tokens: [[Int32]]
        let pixels: MLXArray?
        let grids: [[Int]]
        let arrays: [String: MLXArray]
    }

    let directory: URL
    let items: [Item]
    private var shardName: String?
    private var shard: [String: MLXArray] = [:]

    init(_ directory: URL) throws {
        self.directory = directory
        items = try JSONDecoder().decode(
            [Item].self, from: Data(contentsOf: directory.appendingPathComponent("index.json")))
    }

    mutating func entry(_ item: Item) throws -> Entry {
        if shardName != item.shard {
            shard = try loadArrays(url: directory.appendingPathComponent(item.shard))
            shardName = item.shard
        }
        let n = item.key
        func ints(_ name: String) throws -> [Int32] {
            guard let a = shard["\(n).\(name)"] else { throw ToolError("\(n).\(name) missing") }
            return a.asArray(Int32.self)
        }
        let sequence = D1LFM3Sequence(
            ids: try ints("ids"), stateLength: Int(try ints("state")[0]),
            groups: try ints("groups"),
            slots: try ints("slots").map(Int.init))
        let tokens = try (0 ..< item.slots).map { try ints("tokens.\($0)") }
        var grids = [[Int]]()
        if let g = shard["\(n).grids"] {
            let flat = g.asArray(Int32.self).map(Int.init)
            grids = stride(from: 0, to: flat.count, by: 3).map { Array(flat[$0 ..< $0 + 3]) }
        }
        var arrays = [String: MLXArray]()
        for (key, value) in shard where key.hasPrefix("\(n).") {
            arrays[String(key.dropFirst("\(n).".count))] = value
        }
        return Entry(
            item: item, sequence: sequence, tokens: tokens, pixels: shard["\(n).pixels"],
            grids: grids, arrays: arrays)
    }
}

/// Every image's tokens, in placeholder order.
func imageFeatures(_ engine: D1LFM3Engine, _ entry: EvalSet.Entry) throws -> MLXArray? {
    guard let pixels = entry.pixels else { return nil }
    var start = 0
    var features = [MLXArray]()
    for grid in entry.grids {
        let count = grid[0] * grid[1] * grid[2]
        features.append(
            try engine.imageFeatures(
                pixels[start ..< start + count], rows: grid[1], columns: grid[2]))
        start += count
    }
    return concatenated(features)
}

struct Record: Encodable {
    let key: Int
    let ms: Double
    let visionMs: Double
    let length: Int
    let scores: [[Float]]
}

/// `run --model DIR --data DIR --out FILE.jsonl [--dtype] [--pad N] [--hidden FILE] [--limit N]
/// [--hf-features]`: option-token scores per slot for every sequence.
func runSet(_ arguments: Arguments) throws {
    let start = DispatchTime.now().uptimeNanoseconds
    let engine = try D1LFM3Engine.load(
        directory: try arguments.url("model"), dtype: arguments.dtype)
    log(
        "loaded \(engine.directory.lastPathComponent) dtype \(engine.dtype) in \(Int(milliseconds(since: start))) ms"
    )
    var set = try EvalSet(try arguments.url("data"))
    let out = try arguments.url("out")
    FileManager.default.createFile(atPath: out.path, contents: nil)
    let handle = try FileHandle(forWritingTo: out)
    let encoder = JSONEncoder()
    let pad = arguments.int("pad")
    var saved = [String: MLXArray]()
    var items = set.items
    if let limit = arguments.int("limit") { items = Array(items.prefix(limit)) }
    var times = [Double]()
    for (i, item) in items.enumerated() {
        let entry = try set.entry(item)
        let t0 = DispatchTime.now().uptimeNanoseconds
        var features = try imageFeatures(engine, entry)
        if arguments.flags.contains("hf-features"), let hf = entry.arrays["hf_features"] {
            features = hf
        }
        if let features { eval(features) }
        let visionMs = milliseconds(since: t0)
        let t1 = DispatchTime.now().uptimeNanoseconds
        let length = pad.map { max($0, entry.sequence.ids.count) }
        let hidden = try engine.hidden(entry.sequence, features: features, paddedLength: length)
        eval(hidden)
        let ms = milliseconds(since: t1)
        let scores = entry.tokens.enumerated().map { k, tokens in
            engine.scores(hidden[k ..< k + 1], tokens: tokens)[0]
        }
        eval(scores)
        if arguments.options["hidden"] != nil {
            saved["\(item.key).hidden"] = hidden
            if let features { saved["\(item.key).features"] = features.asType(.float32) }
        }
        let record = Record(
            key: item.key, ms: ms, visionMs: visionMs, length: length ?? entry.sequence.ids.count,
            scores: scores.map { $0.asArray(Float.self) })
        handle.write(try encoder.encode(record))
        handle.write(Data("\n".utf8))
        times.append(ms + visionMs)
        if (i + 1) % 100 == 0 {
            let sorted = times.sorted()
            log("\(i + 1)/\(items.count) median \(Int(sorted[sorted.count / 2])) ms")
        }
    }
    try handle.close()
    if arguments.options["hidden"] != nil {
        try save(arrays: saved, url: try arguments.url("hidden"))
    }
    let sorted = times.sorted()
    log(
        "done \(items.count) sequences; median \(sorted.isEmpty ? 0 : sorted[sorted.count / 2]) ms; peak \(Memory.peakMemory / 1_048_576) MB"
    )
}
