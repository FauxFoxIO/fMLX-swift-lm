// Copyright © 2026 Faux Fox, Inc.
// Port of the d1-omni reference `modeling_d1.py` (LiquidAI/d1-omni-600M) and MLX-VLM `d1_omni`.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

final class D1OmniNetwork: Module {
    let configuration: D1OmniConfiguration

    @ModuleInfo var encoder: D1OmniTrunk
    @ModuleInfo var head: D1OmniDecisionHead

    /// The trunk and decision head. The vision and audio towers load separately, on demand.
    init(_ configuration: D1OmniConfiguration) {
        self.configuration = configuration
        _encoder.wrappedValue = D1OmniTrunk(configuration.text)
        _head.wrappedValue = D1OmniDecisionHead(
            width: configuration.text.hiddenSize, layers: configuration.headLayers)
    }

    var dtype: DType { encoder.embeddingNorm.weight.dtype }

    /// One question row: an optional media prefix [P, D], then the text IDs.
    struct Row {
        let prefix: MLXArray?
        let ids: [Int]
        let markers: [Int]
        let kind: D1OmniQuestion.Kind
        var length: Int { (prefix?.dim(0) ?? 0) + ids.count }
    }

    /// Option logits for right-padded rows, in each row's marker order.
    func logits(_ rows: [Row]) -> [[Float]] {
        let dtype = dtype
        var sequences = [MLXArray]()
        for row in rows {
            let text = encoder.embedTokens(MLXArray(row.ids.map(Int32.init))).asType(dtype)
            sequences.append(row.prefix.map { concatenated([$0.asType(dtype), text]) } ?? text)
        }
        let longest = sequences.map { $0.dim(0) }.max() ?? 0
        let h = stacked(
            sequences.map { padded($0, widths: [[0, longest - $0.dim(0)], 0]) })
        let pad =
            MLXArray(Int32(0) ..< Int32(longest))[.newAxis]
            .< MLXArray(sequences.map { Int32($0.dim(0)) })[0..., .newAxis]
        let prefixes = rows.map { $0.prefix?.dim(0) ?? 0 }
        let hidden = encoder(h, pad: pad, prefix: prefixes)

        let lengths = rows.map(\.ids.count)
        let textLength = lengths.max() ?? 0
        let positions = MLXArray(Int32(0) ..< Int32(textLength))[.newAxis]
        let textPad = positions .< MLXArray(lengths.map(Int32.init))[0..., .newAxis]
        let index = minimum(
            MLXArray(prefixes.map(Int32.init))[0..., .newAxis] + positions, Int32(longest - 1))
        var text = takeAlong(
            hidden,
            broadcast(index[.ellipsis, .newAxis], to: [rows.count, textLength, hidden.dim(-1)]),
            axis: 1)
        text = MLX.where(textPad[.ellipsis, .newAxis], text, MLXArray(0, dtype: text.dtype))

        let k = rows.map(\.markers.count).max() ?? 0
        let markers = MLXArray(
            rows.flatMap { row in
                (row.markers + Array(repeating: 0, count: k - row.markers.count)).map(Int32.init)
            },
            [rows.count, k])
        let markerMask =
            MLXArray(Int32(0) ..< Int32(k))[.newAxis]
            .< MLXArray(rows.map { Int32($0.markers.count) })[0..., .newAxis]
        let types = MLXArray(rows.map { Int32($0.kind.rawValue) })
        let logits = head(
            text, pad: textPad, markers: markers, markerMask: markerMask, types: types)
        eval(logits)
        let values = logits.asArray(Float.self)
        return rows.enumerated().map { i, row in
            Array(values[(i * k) ..< (i * k + row.markers.count)])
        }
    }

    /// Torch checkpoint names and layouts mapped onto this module; MLX exports pass through.
    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        let audioRenames = [
            ("encoder.pre_encode.conv.", "encoder.subsampling.layers."),
            ("encoder.pre_encode.out.", "encoder.subsampling.linear."),
            ("self_attn.linear_q.", "self_attn.q_proj."),
            ("self_attn.linear_k.", "self_attn.k_proj."),
            ("self_attn.linear_v.", "self_attn.v_proj."),
            ("self_attn.linear_out.", "self_attn.o_proj."),
            ("self_attn.linear_pos.", "self_attn.relative_k_proj."),
            ("self_attn.pos_bias_u", "self_attn.bias_u"),
            ("self_attn.pos_bias_v", "self_attn.bias_v"),
            ("conv.batch_norm.", "conv.norm."),
        ]
        var result = [String: MLXArray]()
        for (original, originalValue) in weights {
            var key = original
            var value = originalValue
            if key.hasSuffix("num_batches_tracked") || key.hasSuffix("position_ids") { continue }
            if key.hasPrefix("audio.") {
                for (old, new) in audioRenames {
                    key = key.replacingOccurrences(of: old, with: new)
                }
                if value.ndim == 4, value.dim(2) == value.dim(3) {
                    value = value.transposed(0, 2, 3, 1)
                } else if value.ndim == 3, value.dim(key.contains("pointwise") ? -1 : 1) == 1 {
                    value = value.transposed(0, 2, 1)
                }
                result[key] = value
                continue
            }
            key = key.replacingOccurrences(of: "vision.tower.vision_model.", with: "vision.tower.")
            if key.hasSuffix(".conv.conv.weight"), value.ndim == 3, value.dim(-1) > value.dim(1) {
                value = value.transposed(0, 2, 1)
            }
            if key.hasPrefix("head.head."), let range = key.range(of: ".self_attn.in_proj_") {
                let prefix = key[..<range.lowerBound]
                let suffix = key[range.upperBound...]
                let parts = split(value, parts: 3, axis: 0)
                for (name, part) in zip(["query_proj", "key_proj", "value_proj"], parts) {
                    result["\(prefix).attention.\(name).\(suffix)"] = part
                }
                continue
            }
            if key.hasPrefix("head.head.") {
                key = key.replacingOccurrences(
                    of: ".self_attn.out_proj.", with: ".attention.out_proj.")
            }
            if key.hasPrefix("head.scorer."), !key.hasPrefix("head.scorer.layers.") {
                key = key.replacingOccurrences(of: "head.scorer.", with: "head.scorer.layers.")
            }
            result[key] = value
        }
        return result
    }
}

/// The parts of a d1-omni checkpoint, each loaded into its own module.
enum D1OmniPart: CaseIterable {
    case text, vision, audio

    var prefix: String {
        switch self {
        case .text: ""
        case .vision: "vision."
        case .audio: "audio."
        }
    }

    /// This part's tensors from sanitized checkpoint weights, named relative to its module.
    func weights(_ sanitized: [String: MLXArray]) -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for (key, value) in sanitized {
            let owner: D1OmniPart =
                key.hasPrefix("vision.") ? .vision : key.hasPrefix("audio.") ? .audio : .text
            if owner == self { result[String(key.dropFirst(prefix.count))] = value }
        }
        return result
    }
}

/// Reads a part of the checkpoint when a request first needs it.
struct D1OmniWeightSource: Sendable {
    let files: [URL]
    let quantization: D1OmniConfiguration.Quantization?
    let precision: DecisionPrecision

    func load(_ part: D1OmniPart, into module: Module) throws {
        // Safetensors arrays load lazily, so tensors of other parts are never read.
        let weights = part.weights(D1OmniNetwork.sanitize(try DecisionWeights.load(files)))
        try DecisionWeights.apply(
            weights, to: module, groupSize: quantization?.groupSize, bits: quantization?.bits,
            mode: quantization?.mode, precision: precision)
    }
}

/// A loaded d1-omni decision model. Text, image, and audio requests share the trunk weights.
public final class D1OmniModel: DecisionModel {
    public let artifact: DecisionArtifact
    public let configuration: D1OmniConfiguration
    /// Bytes of the checkpoint files, for residency budgets.
    public let weightBytes: Int64
    public let modalities: Set<DecisionModality> = [.text, .image, .audio]
    /// Padded positions per forward batch, media prefixes included.
    public let tokenBudget: Int
    /// Positions one question row may read, media included.
    public let maximumLength: Int

    private let state: SerialAccessContainer<D1OmniRuntime>

    fileprivate init(
        _ runtime: consuming D1OmniRuntime, artifact: DecisionArtifact, bytes: Int64,
        tokenBudget: Int
    ) {
        configuration = runtime.network.configuration
        maximumLength = runtime.maximumLength
        self.artifact = artifact
        weightBytes = bytes
        self.tokenBudget = tokenBudget
        state = SerialAccessContainer(consume runtime)
    }

    public func decide(_ requests: [DecisionRequest]) async throws -> [DecisionResult] {
        let artifact = artifact
        let budget = tokenBudget
        return try await state.read { runtime in
            try await runtime.decide(requests, tokenBudget: budget, artifact: artifact)
        }
    }

    /// Releases the image or audio tower; the next request that needs it reads it again.
    /// The text trunk stays loaded for the model's lifetime.
    public func unload(_ modalities: Set<DecisionModality>) async {
        await state.read { $0.unload(modalities) }
    }

    public var loadedModalities: Set<DecisionModality> {
        get async { await state.read { $0.loadedModalities } }
    }
}

public enum D1OmniLoader {
    /// Loads a local `d1_omni` snapshot: an MLX export or the original PyTorch checkpoint.
    /// The trunk loads now; the image and audio towers load with their first request unless
    /// `preload` names them. `maximumLength` lowers the checkpoint's 16,384-position row limit
    /// (media included); states are cut to fit, as the reference cuts them.
    public static func load(
        from directory: URL, artifact: DecisionArtifact, precision: DecisionPrecision = .checkpoint,
        tokenBudget: Int = 65536, maximumLength: Int? = nil, preload: Set<DecisionModality> = []
    ) async throws -> D1OmniModel {
        guard !artifact.identifier.isEmpty, !artifact.revision.isEmpty, tokenBudget >= 64,
            (maximumLength ?? 128) >= 128
        else {
            throw DecisionError.invalidConfiguration(
                "provide an artifact identity, a positive budget, and a length of at least 128")
        }
        let configuration = try JSONDecoder().decode(
            D1OmniConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        try configuration.validate()
        let tokenizer = try await NativeDecisionTokenizer.load(from: directory)
        let special = try D1OmniSpecialTokens(tokenizer: tokenizer, bos: configuration.bosTokenID)
        let files = try DecisionWeights.files(in: directory)
        let source = D1OmniWeightSource(
            files: files, quantization: configuration.quantization, precision: precision)
        let network = D1OmniNetwork(configuration)
        try source.load(.text, into: network)
        let runtime = D1OmniRuntime(
            network: network, tokenizer: tokenizer, special: special, source: source,
            maximumLength: maximumLength)
        if preload.contains(.image) { _ = try runtime.visionTower() }
        if preload.contains(.audio) { _ = try runtime.audioEncoder() }
        return D1OmniModel(
            runtime, artifact: artifact, bytes: DecisionWeights.byteCount(files),
            tokenBudget: tokenBudget)
    }
}

final class D1OmniRuntime {
    let network: D1OmniNetwork
    let tokenizer: any DecisionTokenizer
    let special: D1OmniSpecialTokens
    let source: D1OmniWeightSource
    /// Positions one row may read, media included.
    let maximumLength: Int
    private var vision: D1OmniVision?
    private var audio: D1OmniAudio?

    init(
        network: D1OmniNetwork, tokenizer: any DecisionTokenizer, special: D1OmniSpecialTokens,
        source: D1OmniWeightSource, maximumLength: Int? = nil
    ) {
        self.network = network
        self.tokenizer = tokenizer
        self.special = special
        self.source = source
        self.maximumLength = min(
            network.configuration.maxLength, maximumLength ?? network.configuration.maxLength)
    }

    func visionTower() throws -> D1OmniVision {
        if let vision { return vision }
        let tower = D1OmniVision(network.configuration)
        try source.load(.vision, into: tower)
        vision = tower
        return tower
    }

    func audioEncoder() throws -> D1OmniAudio {
        if let audio { return audio }
        let encoder = D1OmniAudio(network.configuration)
        try source.load(.audio, into: encoder)
        audio = encoder
        return encoder
    }

    func unload(_ modalities: Set<DecisionModality>) {
        if modalities.contains(.image) { vision = nil }
        if modalities.contains(.audio) { audio = nil }
    }

    var loadedModalities: Set<DecisionModality> {
        var result: Set<DecisionModality> = [.text]
        if vision != nil { result.insert(.image) }
        if audio != nil { result.insert(.audio) }
        return result
    }

    private struct Pending {
        let row: D1OmniNetwork.Row
        let question: DecisionQuestion
        let parsed: D1OmniQuestion
        let calibrate: Bool
    }

    func decide(_ requests: [DecisionRequest], tokenBudget: Int, artifact: DecisionArtifact)
        async throws -> [DecisionResult]
    {
        let configuration = network.configuration
        var pending = [[Pending]]()
        var read = [Int]()
        for request in requests {
            try Task.checkCancellation()
            guard !request.questions.isEmpty else {
                throw DecisionError.invalidInput("a request needs at least one question")
            }
            for question in request.questions { try question.validate() }
            guard request.images.isEmpty || request.audio == nil else {
                throw DecisionError.invalidInput("a request carries images or audio, not both")
            }
            var prefix: MLXArray?
            var maxLength = maximumLength
            var mode = D1OmniMediaMode.text
            var state = request.state ?? ""
            if !request.images.isEmpty {
                let prepared = try request.images.map {
                    try D1OmniImageProcessor.prepare(DecisionMedia.rgb($0))
                }
                prefix = try visionTower()(prepared)
                maxLength = configuration.imageTextLength
                mode = .image
            } else if let audio = request.audio {
                let samples = try await DecisionMedia.waveform(
                    audio, maximumSeconds: D1OmniMelFrontend.maximumSeconds)
                prefix = try audioEncoder()(samples)
                maxLength = configuration.audioTextLength
                mode = .audio
                state = request.state ?? "{}"
            }
            if let prefix { eval(prefix) }
            let media = prefix?.dim(0) ?? 0
            maxLength = min(maxLength, maximumLength - media)
            guard maxLength >= 64 else {
                throw DecisionError.contextOverflow(
                    "the media take \(media) of the \(maximumLength) positions; send fewer images"
                )
            }
            var rows = [Pending]()
            var tokens = 0
            for question in request.questions {
                let parsed = D1OmniQuestion(question)
                let (ids, markers) = try D1OmniPrompt.encode(
                    tokenizer: tokenizer, special: special, state: state, question: parsed,
                    maxLength: maxLength, mode: mode)
                tokens += media + ids.count
                rows.append(
                    Pending(
                        row: .init(prefix: prefix, ids: ids, markers: markers, kind: parsed.kind),
                        question: question, parsed: parsed, calibrate: mode == .text))
            }
            pending.append(rows)
            read.append(tokens)
        }

        let flat = pending.flatMap { $0 }
        let logits = try run(flat.map(\.row), tokenBudget: tokenBudget)
        var results = [DecisionResult]()
        var offset = 0
        for (rows, tokens) in zip(pending, read) {
            var distributions = [DecisionDistribution]()
            for (i, entry) in rows.enumerated() {
                let z = logits[offset + i].map(Double.init)
                guard z.allSatisfy(\.isFinite) else { throw DecisionError.nonFiniteOutput }
                let temperature = entry.calibrate ? configuration.temperature(for: entry.parsed) : 1
                var probabilities = softmax(z.map { $0 / temperature })
                var ordered = z
                if entry.parsed.kind == .noul {
                    // The model reads a yes/no question as [false, true].
                    probabilities.reverse()
                    ordered.reverse()
                }
                distributions.append(
                    DecisionDistribution(
                        questionID: entry.question.id, outcomes: entry.question.outcomes,
                        probabilities: probabilities, logits: ordered, temperature: temperature))
            }
            offset += rows.count
            results.append(
                DecisionResult(
                    distributions: distributions, inputTokens: tokens, artifact: artifact))
        }
        return results
    }

    /// Rows -> logits, batched longest first under a padded-token budget.
    func run(_ rows: [D1OmniNetwork.Row], tokenBudget: Int) throws -> [[Float]] {
        var results = [[Float]?](repeating: nil, count: rows.count)
        var order = rows.indices.sorted { rows[$0].length > rows[$1].length }
        while !order.isEmpty {
            try Task.checkCancellation()
            let longest = rows[order[0]].length
            var size = 1
            while size < order.count, (size + 1) * longest <= tokenBudget { size += 1 }
            let batch = Array(order.prefix(size))
            order.removeFirst(size)
            for (index, logits) in zip(batch, network.logits(batch.map { rows[$0] })) {
                results[index] = logits
            }
        }
        return results.map { $0! }
    }
}
