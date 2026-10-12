// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

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

final class EmbeddingGemma2WeightStore {
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

    private struct Tensor {
        let rawName: String
        let name: String
        let url: URL
        let shape: [Int]
        let dtype: String
        let bytes: UInt64
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

    private let tensors: [String: Tensor]
    private let fileIdentities: [URL: FileIdentity]
    private let quantization: BaseConfiguration.PerLayerQuantization?
    let precision: EmbeddingGemma2Precision
    let resources: EmbeddingGemma2Resources

    init(
        directory: URL, configuration: EmbeddingGemma2Configuration,
        base: BaseConfiguration, precision: EmbeddingGemma2Precision,
        limits: EmbeddingGemma2ResourceLimits
    ) throws {
        guard directory.isFileURL, (1 ... 8192).contains(limits.maximumTokens),
            limits.maximumCheckpointBytes > 0
        else {
            throw EmbeddingGemma2Error.invalidConfiguration(
                "invalid local directory or resource limits")
        }
        self.precision = precision
        quantization = base.perLayerQuantization
        let files = try Self.files(directory)
        var found = [String: Tensor]()
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
                let shape = Self.canonicalShape(entry.shape, name: name)
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
                guard found[name] == nil else {
                    throw EmbeddingGemma2Error.invalidWeights("sanitizer collision at \(name)")
                }
                let modality = try Self.modality(name)
                byModality[modality, default: 0] += bytes
                found[name] = Tensor(
                    rawName: rawName, name: name, url: url,
                    shape: shape, dtype: entry.dtype, bytes: bytes)
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
        tensors = found
        fileIdentities = identities
        resources = EmbeddingGemma2Resources(
            checkpointBytes: total,
            weightBytesByModality: byModality, maximumTokens: limits.maximumTokens)
        // These modules are lazy shape descriptors; validation does not evaluate random weights.
        try validate(EmbeddingGemma2TextComponent(configuration), modality: .text)
        try validate(EmbeddingGemma2VisionComponent(configuration), modality: .vision)
        try validate(EmbeddingGemma2AudioComponent(configuration), modality: .audio)
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

    private static func canonicalShape(_ shape: [Int], name: String) -> [Int] {
        if name.contains("subsample_conv_projection"), name.hasSuffix("conv.weight"),
            shape.count == 4
        {
            let channels = name.contains(".layer0.") ? 1 : 128
            if shape[3] != channels { return [shape[0], shape[2], shape[3], shape[1]] }
        }
        if name.hasSuffix("depthwise_conv1d.weight"), shape.count == 3, shape[2] != 1 {
            return [shape[0], shape[2], shape[1]]
        }
        return shape
    }

    private static func modality(_ name: String) throws -> EmbeddingGemma2Modality {
        if name.hasPrefix("language_model.") { return .text }
        if name.hasPrefix("vision_tower.") || name.hasPrefix("embed_vision.") { return .vision }
        if name.hasPrefix("audio_tower.") || name.hasPrefix("embed_audio.") { return .audio }
        throw EmbeddingGemma2Error.invalidWeights("unexpected tensor \(name)")
    }

    private func selected(_ modality: EmbeddingGemma2Modality) -> [String: Tensor] {
        tensors.filter { (try? Self.modality($0.key)) == modality }
    }

    private func configureQuantization(_ module: Module) throws {
        for path in module.leafModules().flattened().map({ $0.0 })
        where tensors["\(path).scales"] != nil {
            guard quantizationForPath(path) != nil
            else {
                throw EmbeddingGemma2Error.invalidWeights(
                    "missing quantization configuration for \(path)")
            }
        }
        quantize(model: module) { path, _ in
            guard self.tensors["\(path).scales"] != nil else { return nil }
            return self.quantizationForPath(path)?.asTuple
        }
    }

    private func quantizationForPath(_ path: String) -> BaseConfiguration.Quantization? {
        guard let quantization else { return nil }
        if quantization.perLayerQuantization[path] != nil {
            return quantization.quantization(layer: path)
        }
        if quantization.perLayerQuantization["model.\(path)"] != nil {
            return quantization.quantization(layer: "model.\(path)")
        }
        return quantization.quantization
    }

    private func validate(_ module: Module, modality: EmbeddingGemma2Modality) throws {
        try configureQuantization(module)
        let expected = Dictionary(uniqueKeysWithValues: module.parameters().flattened())
        let provided = selected(modality)
        let missing = Set(expected.keys).subtracting(provided.keys).sorted()
        let extra = Set(provided.keys).subtracting(expected.keys).sorted()
        guard missing.isEmpty, extra.isEmpty else {
            throw EmbeddingGemma2Error.invalidWeights(
                "\(modality.rawValue): missing \(missing), unused \(extra)")
        }
        for (name, parameter) in expected {
            guard let tensor = provided[name], parameter.shape == tensor.shape,
                (parameter.dtype == .uint32) == (tensor.dtype == "U32")
            else {
                throw EmbeddingGemma2Error.invalidWeights(
                    "shape or packed dtype mismatch at \(name)")
            }
        }
    }

    func load(_ module: Module, modality: EmbeddingGemma2Modality) throws {
        try configureQuantization(module)
        let selected = selected(modality)
        var weights = [String: MLXArray]()
        let urls = Set(selected.values.map(\.url)).sorted(by: { $0.path < $1.path })
        for url in urls {
            try Task.checkCancellation()
            try verifyFile(url)
            // MLX reads tensor payloads lazily; unloaded towers never get evaluated.
            let (arrays, _) = try loadArraysAndMetadata(url: url, stream: .cpu)
            for tensor in selected.values where tensor.url == url {
                guard var array = arrays[tensor.rawName] else {
                    throw EmbeddingGemma2Error.invalidWeights("missing tensor \(tensor.rawName)")
                }
                guard [.bfloat16, .float32, .uint32].contains(array.dtype) else {
                    throw EmbeddingGemma2Error.invalidWeights(
                        "checkpoint dtype changed at \(tensor.rawName)")
                }
                if array.shape != tensor.shape {
                    array =
                        array.ndim == 4 ? array.transposed(0, 2, 3, 1) : array.transposed(0, 2, 1)
                }
                if array.dtype != .uint32 { array = array.asType(precision.dtype) }
                weights[tensor.name] = array
            }
        }
        try module.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        module.train(false)
        try checkedEval(module)
        for url in urls { try verifyFile(url) }
    }

    private func verifyFile(_ url: URL) throws {
        guard let expected = fileIdentities[url], try FileIdentity(url) == expected else {
            throw EmbeddingGemma2Error.invalidWeights(
                "checkpoint changed; reload the embedding container")
        }
    }
}
