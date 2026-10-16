// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXLMCommon

/// Limits apply to the complete installed checkpoint, including unloaded media towers.
public struct EmbeddingGemma2ResourceLimits: Sendable {
    public let maximumCheckpointBytes: UInt64
    public let maximumTokens: Int

    public init(maximumCheckpointBytes: UInt64 = 4 * 1024 * 1024 * 1024, maximumTokens: Int = 8192)
    {
        self.maximumCheckpointBytes = maximumCheckpointBytes
        self.maximumTokens = maximumTokens
    }
}

public struct EmbeddingGemma2Resources: Sendable {
    public let checkpointBytes: UInt64
    public let weightBytesByModality: [EmbeddingGemma2Modality: UInt64]
    public let maximumTokens: Int
}

public enum EmbeddingGemma2Precision: String, Codable, Sendable {
    case bfloat16, float32

    var dtype: DType { self == .bfloat16 ? .bfloat16 : .float32 }
}

/// Header-only accounting of the installed checkpoint, read before any tensor loads.
final class EmbeddingGemma2CheckpointInventory: Sendable {
    private struct FileIdentity: Equatable {
        let size: UInt64
        let modified: Date
        let inode: UInt64

        init(_ url: URL) throws {
            let attributes = try FileManager.default.attributesOfItem(
                atPath: url.resolvingSymlinksInPath().path)
            guard let size = attributes[.size] as? NSNumber,
                let modified = attributes[.modificationDate] as? Date,
                let inode = attributes[.systemFileNumber] as? NSNumber
            else {
                throw EmbeddingGemma2Error.invalidWeights(
                    "cannot establish checkpoint file identity")
            }
            self.size = size.uint64Value
            self.modified = modified
            self.inode = inode.uint64Value
        }
    }

    private struct HeaderTensor: Decodable {
        let dtype: String
        let shape: [Int]
        let offsets: [UInt64]
        enum CodingKeys: String, CodingKey {
            case dtype, shape
            case offsets = "data_offsets"
        }
    }

    private struct Index: Decodable {
        let weights: [String: String]
        enum CodingKeys: String, CodingKey { case weights = "weight_map" }
    }

    private let fileIdentities: [URL: FileIdentity]
    let resources: EmbeddingGemma2Resources
    /// Safetensors dtypes of the floating tensors of each modality.
    private let floatingDTypes: [EmbeddingGemma2Modality: Set<String>]

    init(directory: URL, limits: EmbeddingGemma2ResourceLimits) throws {
        guard directory.isFileURL, (1 ... 8192).contains(limits.maximumTokens),
            limits.maximumCheckpointBytes > 0
        else {
            throw EmbeddingGemma2Error.invalidConfiguration(
                "invalid local directory or resource limits")
        }
        let files = try Self.files(directory)
        var names = Set<String>()
        var floating = [EmbeddingGemma2Modality: Set<String>]()
        var identities = [URL: FileIdentity]()
        var rawFiles = [String: String]()
        var total: UInt64 = 0
        var byModality = [EmbeddingGemma2Modality: UInt64]()
        for url in files.urls {
            let identity = try FileIdentity(url)
            for (rawName, entry) in try Self.header(url) {
                guard rawFiles.updateValue(url.lastPathComponent, forKey: rawName) == nil else {
                    throw EmbeddingGemma2Error.invalidWeights("duplicate tensor \(rawName)")
                }
                guard let name = Self.canonicalName(rawName) else { continue }
                let bytes = entry.offsets[1] - entry.offsets[0]
                let (sum, overflow) = total.addingReportingOverflow(bytes)
                guard !overflow, sum <= limits.maximumCheckpointBytes else {
                    throw EmbeddingGemma2Error.invalidWeights(
                        "checkpoint exceeds maximumCheckpointBytes")
                }
                total = sum
                guard ["BF16", "F32", "U32"].contains(entry.dtype) else {
                    throw EmbeddingGemma2Error.invalidWeights(
                        "\(rawName) uses \(entry.dtype); only BF16/FP32 and packed U32 are supported"
                    )
                }
                guard names.insert(name).inserted else {
                    throw EmbeddingGemma2Error.invalidWeights("sanitizer collision at \(name)")
                }
                let modality = try Self.modality(name)
                byModality[modality, default: 0] += bytes
                if entry.dtype != "U32" { floating[modality, default: []].insert(entry.dtype) }
            }
            guard try FileIdentity(url) == identity else {
                throw EmbeddingGemma2Error.invalidWeights(
                    "checkpoint changed during header validation")
            }
            identities[url] = identity
        }
        if let index = files.index, index.weights != rawFiles {
            throw EmbeddingGemma2Error.invalidWeights(
                "safetensors index does not match shard headers")
        }
        fileIdentities = identities
        floatingDTypes = floating
        resources = EmbeddingGemma2Resources(
            checkpointBytes: total,
            weightBytesByModality: byModality, maximumTokens: limits.maximumTokens)
    }

    /// Whether every floating tensor already has the dtype of `precision`.
    func stores(_ precision: EmbeddingGemma2Precision) -> Bool {
        let expected = precision == .bfloat16 ? "BF16" : "F32"
        return floatingDTypes.values.allSatisfy { $0.isSubset(of: [expected]) }
    }

    /// Throws if a checkpoint file changed since the container loaded.
    func verifyFiles() throws {
        for (url, expected) in fileIdentities {
            guard try FileIdentity(url) == expected else {
                throw EmbeddingGemma2Error.invalidWeights(
                    "checkpoint changed; reload the embedding container")
            }
        }
    }

    private static func files(_ directory: URL) throws -> (urls: [URL], index: Index?) {
        let manager = FileManager.default
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if manager.fileExists(atPath: indexURL.path) {
            let index = try JSONDecoder().decode(Index.self, from: Data(contentsOf: indexURL))
            let names = Set(index.weights.values).sorted()
            guard !names.isEmpty,
                names.allSatisfy({
                    !$0.contains("/") && !$0.contains("\\") && $0.hasSuffix(".safetensors")
                })
            else { throw EmbeddingGemma2Error.invalidWeights("invalid shard index") }
            let urls = names.map { directory.appendingPathComponent($0) }
            guard urls.allSatisfy({ manager.fileExists(atPath: $0.path) }) else {
                throw EmbeddingGemma2Error.invalidWeights("missing indexed shard")
            }
            return (urls, index)
        }
        let consolidated = directory.appendingPathComponent("model.safetensors")
        if manager.fileExists(atPath: consolidated.path) { return ([consolidated], nil) }
        let urls = try manager.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter {
                $0.lastPathComponent.hasPrefix("model-") && $0.pathExtension == "safetensors"
            }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !urls.isEmpty else {
            throw EmbeddingGemma2Error.invalidWeights("missing model safetensors")
        }
        return (urls, nil)
    }

    private static func header(_ url: URL) throws -> [String: HeaderTensor] {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        let fileSize = try file.seekToEnd()
        try file.seek(toOffset: 0)
        guard let sizeData = try file.read(upToCount: 8), sizeData.count == 8 else {
            throw EmbeddingGemma2Error.invalidWeights("missing header at \(url.lastPathComponent)")
        }
        let size = sizeData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        guard size > 0, size <= 64 * 1024 * 1024, fileSize >= 8,
            size <= fileSize - 8, let data = try file.read(upToCount: Int(size)),
            data.count == Int(size),
            var json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { throw EmbeddingGemma2Error.invalidWeights("invalid safetensors header") }
        json.removeValue(forKey: "__metadata__")
        let entries = try JSONDecoder().decode(
            [String: HeaderTensor].self,
            from: JSONSerialization.data(withJSONObject: json))
        var previous: UInt64 = 0
        for entry in entries.values.sorted(by: { ($0.offsets.first ?? 0) < ($1.offsets.first ?? 0) }
        ) {
            guard entry.offsets.count == 2, entry.offsets[0] == previous,
                entry.offsets[1] >= entry.offsets[0], entry.offsets[1] <= fileSize - size - 8,
                entry.shape.allSatisfy({ $0 >= 0 })
            else { throw EmbeddingGemma2Error.invalidWeights("invalid tensor byte ranges") }
            let elementBytes: UInt64
            switch entry.dtype {
            case "BF16", "F16": elementBytes = 2
            case "F32", "U32": elementBytes = 4
            default: throw EmbeddingGemma2Error.invalidWeights("unsupported dtype \(entry.dtype)")
            }
            var bytes = elementBytes
            for dimension in entry.shape {
                let multiplied = bytes.multipliedReportingOverflow(by: UInt64(dimension))
                guard !multiplied.overflow else {
                    throw EmbeddingGemma2Error.invalidWeights("tensor size overflow")
                }
                bytes = multiplied.partialValue
            }
            guard bytes == entry.offsets[1] - entry.offsets[0] else {
                throw EmbeddingGemma2Error.invalidWeights(
                    "tensor byte count does not match its shape")
            }
            previous = entry.offsets[1]
        }
        guard previous == fileSize - size - 8 else {
            throw EmbeddingGemma2Error.invalidWeights("unaccounted safetensors payload")
        }
        return entries
    }

    private static func canonicalName(_ name: String) -> String? {
        let name = name.hasPrefix("model.") ? String(name.dropFirst(6)) : name
        if name.contains("rotary_emb.inv_freq") { return nil }
        return name
    }

    private static func modality(_ name: String) throws -> EmbeddingGemma2Modality {
        let text = [
            "language_model.", "embed_tokens.", "embedding_projection.", "ple.", "layers.",
            "norm.",
        ]
        if text.contains(where: name.hasPrefix) { return .text }
        if name.hasPrefix("vision_tower.") || name.hasPrefix("embed_vision.") { return .vision }
        if name.hasPrefix("audio_tower.") || name.hasPrefix("embed_audio.") { return .audio }
        throw EmbeddingGemma2Error.invalidWeights("unexpected tensor \(name)")
    }
}
