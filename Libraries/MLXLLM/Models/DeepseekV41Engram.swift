//
//  DeepseekV41Engram.swift
//  LLM
//
//  DeepSeek-V4.1 Engram memory: each position hashes the n-grams that end there, looks up one
//  row per (n-gram size, head) in a large table, and gates the projected value into each
//  hyper-connection stream. Follows DeepSeek's `inference/engram.py` and `inference/model.py`.
//
//  The two published tables hold about 768M rows (~200 GB in FP8), so they stay on disk:
//  ``DeepseekV41MappedEngramTable`` memory-maps them and copies only the rows a step needs.
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Module

final class DeepseekV41Engram: Module {
    let hc: Int
    let dim: Int
    let eps: Float

    @ModuleInfo(key: "wkv") var wkv: Linear
    @ParameterInfo(key: "q_weight") var qWeight: MLXArray
    @ParameterInfo(key: "k_weight") var kWeight: MLXArray

    init(_ config: DeepseekV41Configuration) {
        hc = config.hcMult
        dim = config.hiddenSize
        eps = config.rmsNormEps
        let columns = (config.engramMaxNgramSize - 1) * config.engramHeads
        _wkv.wrappedValue = Linear(
            columns * config.engramHeadDim, dim * (hc + 1), bias: false)
        _qWeight.wrappedValue = MLXArray.ones([hc, dim])
        _kWeight.wrappedValue = MLXArray.ones([hc, dim])
        super.init()
    }

    /// Adds the gated lookup to streams `[B, L, hc, D]`; `rows` is `[B, L, columns, headDim]`.
    func callAsFunction(_ streams: MLXArray, rows: MLXArray) -> MLXArray {
        let (b, l) = (streams.dim(0), streams.dim(1))
        let kv = wkv(rows.reshaped(b, l, -1))
        let key = kv[.ellipsis, ..<(hc * dim)].reshaped(b, l, hc, dim).asType(.float32)
        let value = kv[.ellipsis, (hc * dim)...].asType(.float32)
        let h = streams.asType(.float32)
        let weight = qWeight.asType(.float32) * kWeight.asType(.float32)
        // Normalized per (token, stream) over `dim`, not jointly over the streams.
        let rstd = rsqrt((h * h).mean(axis: -1) + eps) * rsqrt((key * key).mean(axis: -1) + eps)
        let dot = (h * weight * key).sum(axis: -1) * rstd * Float(pow(Double(dim), -0.5))
        // Signed square root before the sigmoid, as in training.
        let magnitude = sqrt(maximum(abs(dot), Float(1e-6)))
        let gate = sigmoid(MLX.where(dot .< 0, -magnitude, magnitude))
        let out = h + gate.expandedDimensions(axis: -1) * value.expandedDimensions(axis: 2)
        return out.asType(streams.dtype)
    }
}

// MARK: - Hashing

/// Maps token ids to the hash ids of the n-grams ending at each position.
///
/// Token ids first go through the compressed token map, so tokens that normalize alike hash
/// alike. Each position hashes itself with the `maxNgram - 1` tokens before it; positions
/// before the sequence start read the pad token. Every (layer, n-gram size, head) owns a
/// prime-sized bucket range in that layer's table.
public struct DeepseekV41EngramHasher: Sendable {
    public let maxNgram: Int
    public let heads: Int
    /// Bucket modulus per layer, n-gram size major and head minor.
    public let primes: [[Int]]
    /// First row of each bucket range, in the same order as ``primes``.
    public let offsets: [[Int]]
    public let tokenMap: [Int32]
    /// One odd multiplier per (layer, look-back).
    public let multipliers: [[Int64]]
    public let padID: Int32

    public var columns: Int { (maxNgram - 1) * heads }

    public init(
        configuration: DeepseekV41Configuration, tokenMap: [Int32], multipliers: [[Int64]]
    ) throws {
        maxNgram = configuration.engramMaxNgramSize
        heads = configuration.engramHeads
        primes = Self.bucketPrimes(configuration)
        offsets = primes.map { sizes in
            var running = 0
            return sizes.map { size in
                defer { running += size }
                return running
            }
        }
        self.tokenMap = tokenMap
        self.multipliers = multipliers
        let pad = configuration.engramPadTokenID
        guard pad >= 0, pad < tokenMap.count else {
            throw DeepseekV41EngramError.invalid("engram_pad_token_id \(pad) is outside the map")
        }
        padID = tokenMap[pad]

        let layers = configuration.engramLayerIDs.count
        guard multipliers.count == layers, multipliers.allSatisfy({ $0.count == maxNgram }) else {
            throw DeepseekV41EngramError.invalid(
                "multipliers must be [\(layers), \(maxNgram)], got \(multipliers.map(\.count))")
        }
        for (slot, sizes) in primes.enumerated()
        where sizes.reduce(0, +) != configuration.engramNumEmbeddings[slot] {
            throw DeepseekV41EngramError.invalid(
                "Engram layer \(configuration.engramLayerIDs[slot]) buckets sum to "
                    + "\(sizes.reduce(0, +)) rows, config declares "
                    + "\(configuration.engramNumEmbeddings[slot])")
        }
        let distinct = Int(tokenMap.max() ?? -1) + 1
        if configuration.engramCompressedVocabSize > 0,
            distinct != configuration.engramCompressedVocabSize
        {
            throw DeepseekV41EngramError.invalid(
                "token map has \(distinct) compressed ids, config declares "
                    + "\(configuration.engramCompressedVocabSize)")
        }
    }

    /// Primes above `engram_vocab_size - 1`, drawn in order and never reused.
    static func bucketPrimes(_ configuration: DeepseekV41Configuration) -> [[Int]] {
        func isPrime(_ n: Int) -> Bool {
            guard n >= 2 else { return false }
            if n < 4 { return true }
            if n % 2 == 0 { return false }
            var d = 3
            while d * d <= n {
                if n % d == 0 { return false }
                d += 2
            }
            return true
        }
        var seen = Set<Int>()
        return configuration.engramLayerIDs.map { _ in
            var sizes = [Int]()
            for _ in 1 ..< max(configuration.engramMaxNgramSize, 1) {
                var current = configuration.engramVocabSize - 1
                for _ in 0 ..< configuration.engramHeads {
                    current += 1
                    while !isPrime(current) || seen.contains(current) {
                        current += 1
                    }
                    seen.insert(current)
                    sizes.append(current)
                }
            }
            return sizes
        }
    }

    /// The compressed id of `token`; ids outside the map read as the pad token.
    func compressed(_ token: Int32) -> Int32 {
        token >= 0 && Int(token) < tokenMap.count ? tokenMap[Int(token)] : padID
    }

    /// Row ids `[L * layers * columns]` (position major) for `tokens`, which follow `history`,
    /// the compressed ids of the latest positions before `start`.
    func hashes(tokens: [Int32], history: [Int32], start: Int) -> [Int] {
        let layers = multipliers.count
        let context = history + tokens.map(compressed)
        var out = [Int]()
        out.reserveCapacity(tokens.count * layers * columns)
        var window = [Int64](repeating: 0, count: maxNgram)
        for j in 0 ..< tokens.count {
            let position = start + j
            let index = history.count + j
            for shift in 0 ..< maxNgram {
                window[shift] =
                    position < shift ? Int64(padID) : Int64(context[index - shift])
            }
            for layer in 0 ..< layers {
                let m = multipliers[layer]
                var rolling = window[0] &* m[0]
                for i in 1 ..< maxNgram {
                    rolling ^= window[i] &* m[i]
                    for h in 0 ..< heads {
                        let column = (i - 1) * heads + h
                        out.append(
                            Int(rolling % Int64(primes[layer][column])) + offsets[layer][column])
                    }
                }
            }
        }
        return out
    }
}

public enum DeepseekV41EngramError: LocalizedError {
    case missing(String)
    case invalid(String)

    public var errorDescription: String? {
        switch self {
        case .missing(let message), .invalid(let message): message
        }
    }
}

// MARK: - Tables

/// One Engram layer's table.
public protocol DeepseekV41EngramTable: AnyObject, Sendable {
    var rowCount: Int { get }
    var rowWidth: Int { get }
    /// Dequantized float32 rows `[ids.count, rowWidth]`.
    func rows(_ ids: [Int]) -> MLXArray
}

/// How a table is stored.
public enum DeepseekV41EngramTableFormat: Sendable, Equatable {
    /// FP8 E4M3 values with one UE8M0 scale per row and `block` columns.
    case fp8(block: Int)
    /// MLX affine quantization with per-group scales and biases.
    case affine(groupSize: Int, bits: Int)
    /// Unquantized floating point values.
    case float
}

/// A table held in memory, for small models and tests.
public final class DeepseekV41ArrayEngramTable: DeepseekV41EngramTable, @unchecked Sendable {
    let weight: MLXArray
    let scales: MLXArray?
    let biases: MLXArray?
    let format: DeepseekV41EngramTableFormat
    public let rowWidth: Int

    public init(
        weight: MLXArray, scales: MLXArray? = nil, biases: MLXArray? = nil,
        format: DeepseekV41EngramTableFormat
    ) {
        self.weight = weight
        self.scales = scales
        self.biases = biases
        self.format = format
        rowWidth =
            switch format {
            case .affine(_, let bits): weight.dim(1) * 32 / bits
            default: weight.dim(1)
            }
    }

    public var rowCount: Int { weight.dim(0) }

    public func rows(_ ids: [Int]) -> MLXArray {
        let index = MLXArray(ids.map { Int32($0) })
        return deepseekV41DequantizeEngramRows(
            weight.take(index, axis: 0), scales: scales?.take(index, axis: 0),
            biases: biases?.take(index, axis: 0), format: format)
    }
}

func deepseekV41DequantizeEngramRows(
    _ weight: MLXArray, scales: MLXArray?, biases: MLXArray?,
    format: DeepseekV41EngramTableFormat
) -> MLXArray {
    switch format {
    case .fp8(let block):
        let n = weight.dim(0)
        let values = fromFP8(weight, dtype: .float32).reshaped(n, -1, block)
        let scale = pow(MLXArray(Float(2)), scales!.asType(.float32) - 127)
        return (values * scale.expandedDimensions(axis: -1)).reshaped(n, -1)
    case .affine(let groupSize, let bits):
        return dequantized(
            weight, scales: scales!, biases: biases, groupSize: groupSize, bits: bits,
            mode: .affine
        ).asType(.float32)
    case .float:
        return weight.asType(.float32)
    }
}

/// A safetensors tensor's location: absolute byte offset, dtype, and shape.
struct DeepseekV41TensorLocation {
    let file: URL
    let offset: Int
    let dtype: String
    let shape: [Int]

    var rowBytes: Int { shape.dropFirst().reduce(1, *) * Self.itemSize(dtype) }

    static func itemSize(_ dtype: String) -> Int {
        switch dtype {
        case "F64", "I64", "U64": 8
        case "F32", "I32", "U32": 4
        case "F16", "BF16", "I16", "U16": 2
        default: 1
        }
    }

    var mlxType: DType {
        switch dtype {
        case "F32": .float32
        case "F16": .float16
        case "BF16": .bfloat16
        case "U32": .uint32
        case "I32": .int32
        case "I8": .int8
        default: .uint8
        }
    }

    /// The tensors in the safetensors file at `url`, by name.
    static func read(_ url: URL) throws -> [String: DeepseekV41TensorLocation] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
            throw DeepseekV41EngramError.invalid("\(url.lastPathComponent) is not safetensors")
        }
        let length = Int(lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
        guard length > 0, length < 1 << 30, let header = try handle.read(upToCount: length),
            let json = try JSONSerialization.jsonObject(with: header) as? [String: Any]
        else {
            throw DeepseekV41EngramError.invalid("\(url.lastPathComponent) has no header")
        }
        var result = [String: DeepseekV41TensorLocation]()
        for (name, value) in json where name != "__metadata__" {
            guard let entry = value as? [String: Any], let dtype = entry["dtype"] as? String,
                let shape = entry["shape"] as? [Int], let offsets = entry["data_offsets"] as? [Int],
                offsets.count == 2
            else { continue }
            result[name] = DeepseekV41TensorLocation(
                file: url, offset: 8 + length + offsets[0], dtype: dtype, shape: shape)
        }
        return result
    }
}

/// A read-only memory map of a whole file.
final class DeepseekV41MappedFile: @unchecked Sendable {
    let base: UnsafeRawPointer
    let size: Int

    init(_ url: URL) throws {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else {
            throw DeepseekV41EngramError.missing("cannot open \(url.path)")
        }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0 else {
            throw DeepseekV41EngramError.missing("cannot stat \(url.path)")
        }
        size = Int(info.st_size)
        guard let pointer = mmap(nil, size, PROT_READ, MAP_SHARED, fd, 0),
            pointer != MAP_FAILED
        else {
            throw DeepseekV41EngramError.missing("cannot map \(url.path)")
        }
        // Lookups are random rows: read-ahead would only waste page cache.
        madvise(pointer, size, MADV_RANDOM)
        base = UnsafeRawPointer(pointer)
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: base), size)
    }
}

/// A table read from memory-mapped safetensors. Only the looked-up rows are copied, so the
/// table occupies page cache rather than wired GPU memory.
public final class DeepseekV41MappedEngramTable: DeepseekV41EngramTable, @unchecked Sendable {
    private struct Part {
        let file: DeepseekV41MappedFile
        let location: DeepseekV41TensorLocation
    }

    private let weight: Part
    private let scales: Part?
    private let biases: Part?
    let format: DeepseekV41EngramTableFormat
    public let rowCount: Int
    public let rowWidth: Int

    init(
        weight: DeepseekV41TensorLocation, scales: DeepseekV41TensorLocation?,
        biases: DeepseekV41TensorLocation?, format: DeepseekV41EngramTableFormat,
        files: inout [URL: DeepseekV41MappedFile]
    ) throws {
        func part(_ location: DeepseekV41TensorLocation) throws -> Part {
            let file = try files[location.file] ?? DeepseekV41MappedFile(location.file)
            files[location.file] = file
            guard location.offset + location.shape[0] * location.rowBytes <= file.size else {
                throw DeepseekV41EngramError.invalid(
                    "\(location.file.lastPathComponent) is shorter than its header says")
            }
            return Part(file: file, location: location)
        }
        self.weight = try part(weight)
        self.scales = try scales.map(part)
        self.biases = try biases.map(part)
        self.format = format
        rowCount = weight.shape[0]
        rowWidth =
            switch format {
            case .affine(_, let bits): weight.shape[1] * 32 / bits
            default: weight.shape[1]
            }
    }

    private func gather(_ part: Part, _ ids: [Int]) -> MLXArray {
        let rowBytes = part.location.rowBytes
        var data = Data(count: ids.count * rowBytes)
        let source = part.file.base + part.location.offset
        data.withUnsafeMutableBytes { (target: UnsafeMutableRawBufferPointer) in
            let destination = target.baseAddress!
            // Concurrent copies overlap the page faults of cold rows.
            let chunk = 256
            DispatchQueue.concurrentPerform(iterations: (ids.count + chunk - 1) / chunk) { c in
                for i in (c * chunk) ..< min((c + 1) * chunk, ids.count) {
                    precondition(
                        ids[i] >= 0 && ids[i] < rowCount, "Engram row \(ids[i]) out of range")
                    (destination + i * rowBytes).copyMemory(
                        from: source + ids[i] * rowBytes, byteCount: rowBytes)
                }
            }
        }
        let width = rowBytes / DeepseekV41TensorLocation.itemSize(part.location.dtype)
        return MLXArray(data, [ids.count, width], dtype: part.location.mlxType)
    }

    public func rows(_ ids: [Int]) -> MLXArray {
        deepseekV41DequantizeEngramRows(
            gather(weight, ids), scales: scales.map { gather($0, ids) },
            biases: biases.map { gather($0, ids) }, format: format)
    }
}

// MARK: - Memory

/// The Engram hasher and tables of a model, plus the per-step lookup.
public final class DeepseekV41EngramMemory: @unchecked Sendable {
    public static let tablesFileName = "deepseek_v41_engram.safetensors"

    let configuration: DeepseekV41Configuration
    public private(set) var hasher: DeepseekV41EngramHasher?
    public private(set) var tables: [DeepseekV41EngramTable] = []

    init(_ configuration: DeepseekV41Configuration) {
        self.configuration = configuration
    }

    var isEnabled: Bool { !configuration.engramLayerIDs.isEmpty }

    /// Installs the hasher and one table per Engram layer, in `engram_layer_ids` order.
    public func configure(hasher: DeepseekV41EngramHasher, tables: [DeepseekV41EngramTable])
        throws
    {
        guard tables.count == configuration.engramLayerIDs.count else {
            throw DeepseekV41EngramError.invalid(
                "expected \(configuration.engramLayerIDs.count) Engram tables, got \(tables.count)")
        }
        for (slot, table) in tables.enumerated() {
            guard table.rowWidth == configuration.engramHeadDim,
                table.rowCount >= configuration.engramNumEmbeddings[slot]
            else {
                throw DeepseekV41EngramError.invalid(
                    "Engram table \(slot) is [\(table.rowCount), \(table.rowWidth)], expected "
                        + "[\(configuration.engramNumEmbeddings[slot]), "
                        + "\(configuration.engramHeadDim)]")
            }
        }
        self.hasher = hasher
        self.tables = tables
    }

    /// Loads the hash tables written by `scripts/deepseek_v41_engram_tables.py` and maps the
    /// Engram tables from the checkpoint's safetensors files.
    public func load(directory: URL) throws {
        guard isEnabled else { return }
        let sidecar = directory.appending(component: Self.tablesFileName)
        guard FileManager.default.fileExists(atPath: sidecar.path) else {
            throw DeepseekV41EngramError.missing(
                "DeepSeek-V4.1 Engram needs \(Self.tablesFileName) beside the checkpoint; "
                    + "write it with scripts/deepseek_v41_engram_tables.py")
        }
        let arrays = try loadArrays(url: sidecar)
        guard let map = arrays["token_map"], let multipliers = arrays["multipliers"],
            multipliers.ndim == 2
        else {
            throw DeepseekV41EngramError.invalid(
                "\(Self.tablesFileName) lacks token_map or multipliers")
        }
        let width = multipliers.dim(1)
        let flat = multipliers.asType(.int64).asArray(Int64.self)
        let hasher = try DeepseekV41EngramHasher(
            configuration: configuration, tokenMap: map.asType(.int32).asArray(Int32.self),
            multipliers: (0 ..< multipliers.dim(0)).map {
                Array(flat[($0 * width) ..< (($0 + 1) * width)])
            })
        try configure(hasher: hasher, tables: try mapTables(directory: directory))
    }

    private func mapTables(directory: URL) throws -> [DeepseekV41EngramTable] {
        let files = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        ).filter {
            $0.pathExtension == "safetensors" && $0.lastPathComponent != Self.tablesFileName
        }
        var tensors = [String: DeepseekV41TensorLocation]()
        for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            tensors.merge(try DeepseekV41TensorLocation.read(file)) { first, _ in first }
        }
        let quantization = try affineQuantization(directory: directory)
        var mapped = [URL: DeepseekV41MappedFile]()
        return try configuration.engramLayerIDs.map { layer in
            let prefix = ["layers.\(layer).engram.embed", "model.layers.\(layer).engram.embed"]
                .first { tensors["\($0).weight"] != nil }
            guard let prefix, let weight = tensors["\(prefix).weight"] else {
                throw DeepseekV41EngramError.missing("no Engram table for layer \(layer)")
            }
            if let scale = tensors["\(prefix).scale"] {
                guard weight.dtype == "F8_E4M3", weight.shape[1] % scale.shape[1] == 0 else {
                    throw DeepseekV41EngramError.invalid("unsupported Engram table \(weight.dtype)")
                }
                return try DeepseekV41MappedEngramTable(
                    weight: weight, scales: scale, biases: nil,
                    format: .fp8(block: weight.shape[1] / scale.shape[1]), files: &mapped)
            }
            if let scales = tensors["\(prefix).scales"] {
                let (groupSize, bits) = quantization
                guard weight.dtype == "U32" else {
                    throw DeepseekV41EngramError.invalid("unsupported Engram table \(weight.dtype)")
                }
                return try DeepseekV41MappedEngramTable(
                    weight: weight, scales: scales, biases: tensors["\(prefix).biases"],
                    format: .affine(groupSize: groupSize, bits: bits), files: &mapped)
            }
            return try DeepseekV41MappedEngramTable(
                weight: weight, scales: nil, biases: nil, format: .float, files: &mapped)
        }
    }

    /// The default affine settings from `config.json`, for MLX-quantized tables.
    private func affineQuantization(directory: URL) throws -> (Int, Int) {
        let data = try Data(contentsOf: directory.appending(component: "config.json"))
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let quantization = json?["quantization"] as? [String: Any]
        return (quantization?["group_size"] as? Int ?? 64, quantization?["bits"] as? Int ?? 4)
    }

    /// Rows `[B, L, columns, headDim]` for each Engram layer, keyed by layer index. Updates the
    /// token history in `cache`.
    func lookup(_ inputs: MLXArray, cache: DeepseekV41Cache) -> [Int: MLXArray] {
        guard isEnabled else { return [:] }
        guard let hasher, !tables.isEmpty else {
            fatalError(
                "DeepSeek-V4.1 Engram tables are not loaded; load the model from its directory")
        }
        let (b, l) = (inputs.dim(0), inputs.dim(1))
        let tokens = inputs.asType(.int32).asArray(Int32.self)
        if cache.engramHistory.count != b {
            cache.engramHistory = Array(repeating: [], count: b)
        }
        let start = cache.offset
        let layers = tables.count
        let columns = hasher.columns
        var ids = Array(repeating: [Int](), count: layers)
        for row in 0 ..< b {
            let rowTokens = Array(tokens[(row * l) ..< ((row + 1) * l)])
            let history = cache.engramHistory[row]
            let hashes = hasher.hashes(tokens: rowTokens, history: history, start: start)
            for j in 0 ..< l {
                for slot in 0 ..< layers {
                    let base = (j * layers + slot) * columns
                    ids[slot].append(contentsOf: hashes[base ..< (base + columns)])
                }
            }
            let keep = hasher.maxNgram - 1 + cache.rewindLimit
            cache.engramHistory[row] = Array(
                (history + rowTokens.map(hasher.compressed)).suffix(keep))
        }
        var result = [Int: MLXArray]()
        for (slot, layer) in configuration.engramLayerIDs.enumerated() {
            result[layer] = tables[slot].rows(ids[slot]).reshaped(b, l, columns, -1)
        }
        return result
    }
}
