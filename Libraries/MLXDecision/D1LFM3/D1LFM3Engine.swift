// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// One d1 row as the host renders it: token IDs, the state's length, each position's question
/// group, and the answer slot of each question.
public struct D1LFM3Sequence: Sendable {
    public var ids: [Int32]
    public var stateLength: Int
    /// One group per position; positions of one question share a group. Empty means one question
    /// that reads the whole row.
    public var groups: [Int32]
    public var slots: [Int]

    public init(ids: [Int32], stateLength: Int, groups: [Int32], slots: [Int]) {
        self.ids = ids
        self.stateLength = stateLength
        self.groups = groups
        self.slots = slots
    }
}

/// The modules of an `lfm3_vl` checkpoint, named as the MLX export names them.
final class D1LFM3Network: Module {
    @ModuleInfo(key: "language_model") var text: D1LFM3Text
    @ModuleInfo(key: "vision_tower") var visionTower: D1LFM3VisionTower?
    @ModuleInfo(key: "multi_modal_projector") var projector: D1LFM3Projector?

    init(_ configuration: D1LFM3Configuration, vision: Bool) throws {
        _text.wrappedValue = try D1LFM3Text(configuration.text)
        if vision, let config = configuration.vision {
            _visionTower.wrappedValue = try D1LFM3VisionTower(config)
            _projector.wrappedValue = D1LFM3Projector(configuration, vision: config)
        }
    }

    /// Checkpoint names and layouts mapped onto these modules; MLX exports pass through.
    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (original, value) in weights {
            var key = original.hasPrefix("model.") ? String(original.dropFirst(6)) : original
            var value = value
            if key.hasSuffix(".conv.conv.weight"), value.ndim == 3 {
                // [D, 1, k] → tap-major [k, D]
                key = key.replacingOccurrences(of: ".conv.conv.weight", with: ".conv.conv_weight")
                value = value[0..., 0, 0...].transposed(1, 0)
            }
            if key.hasSuffix(".spatial_conv.weight"), value.ndim == 4, value.dim(1) == 1 {
                // [D, 1, kh, kw] → [D, kh, kw, 1]
                value = value.transposed(0, 2, 3, 1)
            }
            if key == "lm_head.weight" { continue }
            result[key] = value
        }
        return result
    }
}

/// Liquid d1-2B (LFM3 text with the LFM3-VL vision tower) as a stateless decision trunk: one
/// forward per row returns the final-norm hidden states at the answer slots. The host computes
/// option logits from them with the tied embedding rows (``readoutRows(_:)``, ``scores(_:tokens:)``).
///
/// An engine is not thread-safe: run one call at a time.
public final class D1LFM3Engine {
    public let configuration: D1LFM3Configuration
    public let directory: URL
    /// The floating-point type of activations and unquantized weights.
    public let dtype: DType
    /// Memory-mapped loads: tensors MLX had to copy, and bytes used in place from the files.
    public internal(set) var copiedTensors: [String] = []
    public internal(set) var mappedBytes = 0
    let network: D1LFM3Network

    init(configuration: D1LFM3Configuration, directory: URL, dtype: DType, network: D1LFM3Network) {
        self.configuration = configuration
        self.directory = directory
        self.dtype = dtype
        self.network = network
    }

    public var readsImages: Bool { network.visionTower != nil }

    /// Evaluates the trunk every this many layers (0: one lazy graph per row). Smaller intervals
    /// lower peak memory.
    public var evaluationInterval: Int {
        get { network.text.evaluationInterval }
        set { network.text.evaluationInterval = newValue }
    }

    /// Loads an MLX export, or the merged PyTorch checkpoint, from a local directory.
    /// `dtype` casts floating-point weights (default: the stored type). `.memoryMapped` uses the
    /// file's pages in place: an export in its own dtype then stays file-backed, outside the
    /// process's footprint.
    public static func load(
        directory: URL, dtype: DType? = nil, vision: Bool = true,
        weightLoading: WeightLoading = .copy
    ) throws -> D1LFM3Engine {
        let configuration = try D1LFM3Configuration.load(directory)
        let network = try D1LFM3Network(configuration, vision: vision)
        let files = try DecisionWeights.files(in: directory)
        let wanted = { (name: String) in vision || name.contains("language_model.") }
        var raw = [String: MLXArray]()
        var copied = [String]()
        var mapped = 0
        switch weightLoading {
        case .copy:
            raw = try DecisionWeights.load(files).filter { wanted($0.key) }
        case .memoryMapped:
            for file in files {
                let loaded = try loadMappedSafetensors(url: file, isRead: wanted)
                raw.merge(loaded.arrays) { a, _ in a }
                copied += loaded.copiedTensorNames
                mapped += loaded.mappedBytes
            }
        }
        let weights = D1LFM3Network.sanitize(raw)
        let stored = weights["language_model.embedding_norm.weight"]?.dtype ?? .bfloat16
        let target = dtype ?? stored
        try apply(weights, to: network, quantization: configuration.quantization, dtype: target)
        let engine = D1LFM3Engine(
            configuration: configuration, directory: directory, dtype: target, network: network)
        engine.copiedTensors = copied
        engine.mappedBytes = mapped
        return engine
    }

    /// Swaps in quantized layers for the paths that ship `.scales`, casts floating-point weights,
    /// and requires an exact match.
    static func apply(
        _ weights: [String: MLXArray], to network: Module,
        quantization: D1LFM3Configuration.Quantization?, dtype: DType
    ) throws {
        if weights.keys.contains(where: { $0.hasSuffix(".scales") }) {
            guard let quantization else {
                throw DecisionError.invalidConfiguration(
                    "the export is quantized but its config has no quantization")
            }
            quantize(model: network) { path, _ in
                guard weights["\(path).scales"] != nil,
                    let setting = quantization.setting(for: path)
                else { return nil }
                return (setting.groupSize, setting.bits, setting.quantizationMode)
            }
        }
        var parameters = weights
        for (key, value) in weights where value.dtype.isFloatingPoint && value.dtype != dtype {
            parameters[key] = value.asType(dtype)
        }
        do {
            try network.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])
        } catch {
            throw DecisionError.invalidWeights(String(describing: error))
        }
        network.train(false)
        eval(network)
    }

    /// One image's tokens: `pixels` [rows × columns, patch values] in merge-block order (the
    /// processor's `pixel_values` for one frame) → [rows × columns / merge², hidden].
    public func imageFeatures(_ pixels: MLXArray, rows: Int, columns: Int) throws -> MLXArray {
        guard let tower = network.visionTower, let projector = network.projector else {
            throw DecisionError.unsupportedInput("this engine was loaded without the vision tower")
        }
        let merge = tower.configuration.mergeSize
        guard rows % merge == 0, columns % merge == 0, pixels.dim(0) == rows * columns else {
            throw DecisionError.invalidInput(
                "\(pixels.dim(0)) patches do not fill a \(rows)×\(columns) grid")
        }
        return projector(tower(pixels.asType(dtype), rows: rows, columns: columns))
    }

    /// Input embeddings [1, L, D]: the token rows, with `features` [image tokens, D] written over the
    /// `imageToken` placeholders in order.
    public func embeddings(_ sequence: D1LFM3Sequence, features: MLXArray? = nil) throws -> MLXArray
    {
        let ids = MLXArray(sequence.ids)
        var h = network.text.embedTokens(ids).asType(dtype)
        let image = Int32(configuration.imageTokenId)
        let positions = sequence.ids.enumerated().filter { $0.element == image }.map {
            Int32($0.offset)
        }
        if !positions.isEmpty || features != nil {
            guard let features, features.dim(0) == positions.count else {
                throw DecisionError.invalidInput(
                    "\(positions.count) image placeholders but \(features?.dim(0) ?? 0) image tokens"
                )
            }
            h[MLXArray(positions)] = features.asType(dtype)
        }
        return h[.newAxis]
    }

    /// The final-norm hidden states at the sequence's answer slots, [K, D] float32 (lazy).
    /// `paddedLength` right-pads the row (padding is masked, so real positions are unchanged).
    public func hidden(
        _ sequence: D1LFM3Sequence, features: MLXArray? = nil, paddedLength: Int? = nil
    ) throws -> MLXArray {
        let length = sequence.ids.count
        guard length > 0, !sequence.slots.isEmpty,
            sequence.slots.allSatisfy({ $0 >= 0 && $0 < length })
        else {
            throw DecisionError.invalidInput("answer slots must fall inside the row")
        }
        guard sequence.groups.isEmpty || sequence.groups.count == length else {
            throw DecisionError.invalidInput("groups must have one entry per position")
        }
        var groups =
            sequence.groups.isEmpty
            ? MLXArray.ones([length], dtype: .int32) : MLXArray(sequence.groups)
        var h = try embeddings(sequence, features: features)
        if let paddedLength, paddedLength > length {
            h = padded(h, widths: [0, [0, paddedLength - length], 0])
            groups = padded(
                groups, widths: [[0, paddedLength - length]], value: MLXArray(Int32(-2)))
        }
        return network.text(
            h, real: length, stateLength: sequence.stateLength, groups: groups,
            slots: MLXArray(sequence.slots.map(Int32.init))
        ).asType(.float32)
    }

    /// The tied head's rows for `tokens`, [T, D] float32 (dequantized when the table is quantized).
    public func readoutRows(_ tokens: [Int32]) -> MLXArray {
        network.text.embedTokens(MLXArray(tokens)).asType(.float32)
    }

    /// `hidden` [K, D] against the readout rows of `tokens`: [K, T] float32 logits.
    public func scores(_ hidden: MLXArray, tokens: [Int32]) -> MLXArray {
        matmul(hidden.asType(.float32), readoutRows(tokens).transposed())
    }
}
