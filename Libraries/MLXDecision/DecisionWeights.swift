// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXNN

enum DecisionWeights {
    private struct Index: Decodable {
        let weightMap: [String: String]
        enum CodingKeys: String, CodingKey { case weightMap = "weight_map" }
    }

    /// The safetensors files of a local snapshot: the index's files, else `model*.safetensors`.
    static func files(in directory: URL) throws -> [URL] {
        guard directory.isFileURL else {
            throw DecisionError.invalidConfiguration("weights need a local snapshot directory")
        }
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: index.path) {
            let map = try JSONDecoder().decode(Index.self, from: Data(contentsOf: index)).weightMap
            let urls = Set(map.values).sorted().map { directory.appendingPathComponent($0) }
            guard !urls.isEmpty,
                urls.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) })
            else {
                throw DecisionError.invalidWeights("the weight index names missing files")
            }
            return urls
        }
        let urls = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )
        .filter { $0.lastPathComponent.hasPrefix("model") && $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !urls.isEmpty else {
            throw DecisionError.invalidWeights(
                "no model*.safetensors in \(directory.lastPathComponent)")
        }
        return urls
    }

    static func load(_ urls: [URL]) throws -> [String: MLXArray] {
        var weights = [String: MLXArray]()
        for url in urls {
            for (key, value) in try loadArrays(url: url) {
                guard weights.updateValue(value, forKey: key) == nil else {
                    throw DecisionError.invalidWeights("tensor \(key) appears twice")
                }
            }
        }
        return weights
    }

    static func byteCount(_ urls: [URL]) -> Int64 {
        urls.reduce(0) { total, url in
            let size =
                (try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? NSNumber
            return total + (size?.int64Value ?? 0)
        }
    }

    /// Quantizes the layers that ship `.scales`, casts on request, then requires an exact match.
    static func apply(
        _ weights: [String: MLXArray], to model: Module, groupSize: Int?, bits: Int?, mode: String?,
        precision: DecisionPrecision
    ) throws {
        if weights.keys.contains(where: { $0.hasSuffix(".scales") }) {
            guard let groupSize, let bits else {
                throw DecisionError.invalidConfiguration(
                    "the checkpoint is quantized but its config has no quantization")
            }
            let quantizationMode: QuantizationMode
            switch mode ?? "affine" {
            case "affine": quantizationMode = .affine
            case "mxfp4": quantizationMode = .mxfp4
            default:
                throw DecisionError.invalidConfiguration(
                    "unsupported quantization mode \(mode ?? "")")
            }
            quantize(model: model) { path, _ in
                weights["\(path).scales"] != nil ? (groupSize, bits, quantizationMode) : nil
            }
        }
        var parameters = weights
        if precision == .float32 {
            for (key, value) in weights where value.dtype == .float16 || value.dtype == .bfloat16 {
                parameters[key] = value.asType(.float32)
            }
        }
        do {
            try model.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])
        } catch {
            throw DecisionError.invalidWeights(String(describing: error))
        }
        model.train(false)
        eval(model)
    }
}
