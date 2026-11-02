// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Writes MLX exports of an `lfm3_vl` d1 checkpoint: renamed and reshaped tensors in one
/// floating-point type, with per-tensor quantization from a recipe.
public enum D1LFM3Export {
    /// Files copied beside the weights when the source has them.
    static let companions = [
        "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "calibration.json",
        "processor_config.json", "LICENSE",
    ]

    public struct Report: Sendable {
        public var tensors: [String: String] = [:]
        public var bytes: Int64 = 0
        /// The activation-scaling exponent chosen for each fold, keyed by its importance layer.
        public var alphas: [String: Float] = [:]
    }

    /// The module paths a recipe may quantize: every linear layer and the token embeddings.
    static func quantizable(_ key: String, _ value: MLXArray) -> Bool {
        key.hasSuffix(".weight") && value.ndim == 2 && !key.contains("norm")
            && value.dim(1) % 32 == 0
    }

    /// Converts `checkpoint` (a merged PyTorch checkpoint or an MLX export) into `output`.
    /// `importance` maps module paths to the mean square of their input channels.
    public static func convert(
        checkpoint: URL, output: URL, dtype: DType, recipe: D1LFM3Recipe?,
        importance: [String: MLXArray] = [:], includeVision: Bool = true
    ) throws -> Report {
        var configuration = try D1LFM3Configuration.load(checkpoint)
        var weights = D1LFM3Network.sanitize(
            try DecisionWeights.load(DecisionWeights.files(in: checkpoint)))
        if !includeVision {
            weights = weights.filter { $0.key.hasPrefix("language_model.") }
        }
        var importance = importance
        var report = Report()
        if let recipe, recipe.scaling == true, !importance.isEmpty {
            report.alphas = D1LFM3Scaling.apply(
                &weights, importance: &importance, configuration: configuration.text,
                recipe: recipe,
                scaleType: dtype)
        }
        var result = [String: MLXArray]()
        var overrides = [String: D1LFM3Configuration.Quantization.Setting?]()
        var used = [D1LFM3Configuration.Quantization.Setting: Int]()
        for key in weights.keys.sorted() {
            var value = weights[key]!
            if value.dtype.isFloatingPoint { value = value.asType(dtype) }
            let path = String(key.dropLast(".weight".count))
            if let recipe, quantizable(key, value), let setting = recipe.setting(for: path) {
                let q = D1LFM3Quantizer.quantize(
                    weights[key]!, setting: setting, method: recipe.method,
                    importance: importance[path], scaleType: dtype)
                result[key] = q.weight
                result["\(path).scales"] = q.scales
                if let biases = q.biases { result["\(path).biases"] = biases }
                eval(q.weight, q.scales)
                report.tensors[path] = "\(setting.mode) \(setting.bits)-bit g\(setting.groupSize)"
                used[setting, default: 0] += value.size
                overrides[path] = setting
            } else {
                result[key] = value
                eval(value)
                report.tensors[path] = "\(dtype)"
            }
        }
        if !used.isEmpty {
            let common = used.max { $0.value < $1.value }!.key
            configuration.quantization = .init(
                defaults: common, overrides: overrides.filter { $0.value != common })
        } else {
            configuration.quantization = nil
        }

        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let weightsURL = output.appendingPathComponent("model.safetensors")
        try saveSafetensorsForMapping(arrays: result, metadata: ["format": "mlx"], url: weightsURL)
        report.bytes =
            ((try? FileManager.default.attributesOfItem(atPath: weightsURL.path)[.size])
            as? NSNumber)?.int64Value ?? 0

        // The source config with the export's quantization; keys this type does not model are kept.
        let source = try Data(contentsOf: checkpoint.appendingPathComponent("config.json"))
        var json = try JSONSerialization.jsonObject(with: source) as? [String: Any] ?? [:]
        json.removeValue(forKey: "quantization")
        json.removeValue(forKey: "auto_map")
        if let quantization = configuration.quantization {
            json["quantization"] = try JSONSerialization.jsonObject(
                with: JSONEncoder().encode(quantization))
        }
        if !includeVision { json.removeValue(forKey: "vision_config") }
        json["torch_dtype"] = String(describing: dtype)
        try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
            .write(to: output.appendingPathComponent("config.json"))
        for name in companions {
            let from = checkpoint.appendingPathComponent(name)
            let to = output.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: from.path), from != to else { continue }
            try? FileManager.default.removeItem(at: to)
            try FileManager.default.copyItem(at: from, to: to)
        }
        return report
    }
}

// MARK: - Calibration

/// A linear layer that adds the square of each input channel to a running sum.
final class D1LFM3ObservedLinear: Linear {
    let record: (MLXArray) -> Void

    init(_ base: Linear, record: @escaping (MLXArray) -> Void) {
        self.record = record
        super.init(weight: base.weight, bias: base.bias)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        record(x)
        return super.callAsFunction(x)
    }
}

extension D1LFM3Engine {
    /// The module paths of every unquantized linear layer.
    public var linearPaths: [String] {
        network.leafModules().flattened().compactMap { path, module in
            module is Linear && !(module is QuantizedLinear) ? path : nil
        }
    }

    /// Runs `body` with every linear layer observed, and returns each layer's mean squared input
    /// channel [I] (an importance matrix) keyed by module path.
    public func collectImportance(_ body: () throws -> Void) rethrows -> [String: MLXArray] {
        var sums = [String: MLXArray]()
        var counts = [String: Int]()
        let originals = network.leafModules().flattened().compactMap {
            path, module -> (String, Linear)? in
            guard let linear = module as? Linear, !(linear is QuantizedLinear) else { return nil }
            return (path, linear)
        }
        let observed = originals.map { path, linear -> (String, Module) in
            (
                path,
                D1LFM3ObservedLinear(linear) { x in
                    let flat = x.reshaped(-1, x.dim(-1)).asType(.float32)
                    let s = (flat * flat).sum(axis: 0)
                    sums[path] = sums[path].map { $0 + s } ?? s
                    counts[path, default: 0] += flat.dim(0)
                    eval(sums[path]!)
                }
            )
        }
        network.update(modules: ModuleChildren.unflattened(observed))
        defer { network.update(modules: ModuleChildren.unflattened(originals.map { ($0, $1) })) }
        try body()
        var result = [String: MLXArray]()
        for (path, sum) in sums {
            result[path] = sum / Float(max(1, counts[path] ?? 1))
        }
        return result
    }

    /// The floating-point weight of the linear layer at `path`.
    public func linearWeight(_ path: String) -> MLXArray? {
        network.leafModules().flattened().first { $0.0 == path }.flatMap {
            ($0.1 as? Linear)?.weight
        }
    }

    /// Replaces linear layers in place: each path maps to a quantized layer or back to a float one.
    public func replaceLinear(_ layers: [String: Linear]) {
        let modules = Dictionary(uniqueKeysWithValues: network.namedModules())
        for (path, layer) in layers {
            guard let dot = path.lastIndex(of: "."), let parent = modules[String(path[..<dot])]
            else { continue }
            parent.update(
                modules: ModuleChildren.unflattened([
                    (String(path[path.index(after: dot)...]), layer)
                ]))
        }
    }

    /// The token embeddings' float weight, for quantization experiments.
    public var embeddingWeight: MLXArray { network.text.embedTokens.weight }

    /// Replaces the token embeddings (float or quantized).
    public func replaceEmbedding(_ embedding: Embedding) {
        network.text.update(modules: ModuleChildren.unflattened([("embed_tokens", embedding)]))
    }
}
