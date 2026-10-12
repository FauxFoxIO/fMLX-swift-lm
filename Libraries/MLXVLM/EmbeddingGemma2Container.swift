// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXLMCommon

/// Standalone, serialized native embeddings. No causal model, chat session, or generation cache.
public final class EmbeddingGemma2Container: Sendable {
    public let artifact: EmbeddingGemma2Artifact
    public let resources: EmbeddingGemma2Resources
    private let state: SerialAccessContainer<EmbeddingGemma2Runtime>

    fileprivate init(_ runtime: consuming EmbeddingGemma2Runtime) {
        artifact = runtime.artifact
        resources = runtime.weights.resources
        state = SerialAccessContainer(consume runtime)
    }

    /// Encodes one ordered unit. Oversized input throws; the caller owns segmentation.
    public func embed(
        _ content: [EmbeddingGemma2Content],
        purpose: EmbeddingGemma2Purpose = .document(),
        dimension: EmbeddingGemma2Dimension = .d768,
        imageSoftTokenBudget: Int = 280
    ) async throws -> EmbeddingGemma2Result {
        try await state.read { runtime in
            try await runtime.embed(
                content, purpose: purpose, dimension: dimension,
                imageSoftTokenBudget: imageSoftTokenBudget)
        }
    }

    public func embed(
        text: String, purpose: EmbeddingGemma2Purpose = .document(),
        dimension: EmbeddingGemma2Dimension = .d768
    ) async throws -> EmbeddingGemma2Result {
        try await embed([.text(text)], purpose: purpose, dimension: dimension)
    }

    /// Releases selected encoder weights; the next matching request reloads them from disk.
    public func unload(
        _ modalities: Set<EmbeddingGemma2Modality> = Set(EmbeddingGemma2Modality.allCases)
    ) async {
        await state.read { $0.unload(modalities) }
    }

    public var loadedModalities: Set<EmbeddingGemma2Modality> {
        get async { await state.read { $0.loadedModalities } }
    }
}

public enum EmbeddingGemma2Loader {
    /// Loads the installed checkpoint tokenizer directly using the package's native tokenizer engine.
    public static func loadContainer(
        from directory: URL,
        artifact: EmbeddingGemma2Artifact,
        precision: EmbeddingGemma2Precision = .bfloat16,
        limits: EmbeddingGemma2ResourceLimits = .init()
    ) async throws -> EmbeddingGemma2Container {
        let tokenizer = try await EmbeddingGemma2Tokenizer.load(from: directory)
        return try loadContainer(
            from: directory, tokenizer: tokenizer, artifact: artifact,
            precision: precision, limits: limits)
    }

    /// Loads tokenizer and text weights from an already installed complete multimodal checkpoint.
    /// `artifact.revision` identifies the actual installed snapshot, not the upstream source revision.
    public static func loadContainer(
        from directory: URL,
        using tokenizerLoader: any TokenizerLoader,
        artifact: EmbeddingGemma2Artifact,
        precision: EmbeddingGemma2Precision = .bfloat16,
        limits: EmbeddingGemma2ResourceLimits = .init()
    ) async throws -> EmbeddingGemma2Container {
        let assets = try EmbeddingGemma2Tokenizer.checkedAssets(directory)
        let tokenizer = try await tokenizerLoader.load(from: directory)
        guard try EmbeddingGemma2Tokenizer.checkedAssets(directory) == assets else {
            throw EmbeddingGemma2Error.invalidConfiguration(
                "tokenizer assets changed during loading")
        }
        return try loadContainer(
            from: directory, tokenizer: EmbeddingGemma2Tokenizer(tokenizer),
            artifact: artifact, precision: precision, limits: limits)
    }

    private static func loadContainer(
        from directory: URL, tokenizer: EmbeddingGemma2Tokenizer,
        artifact: EmbeddingGemma2Artifact, precision: EmbeddingGemma2Precision,
        limits: EmbeddingGemma2ResourceLimits
    ) throws -> EmbeddingGemma2Container {
        guard directory.isFileURL, !artifact.identifier.isEmpty, !artifact.revision.isEmpty,
            artifact.revision != "main", artifact.revision != "latest"
        else {
            throw EmbeddingGemma2Error.invalidConfiguration(
                "provide a local snapshot and immutable artifact revision")
        }
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(EmbeddingGemma2Configuration.self, from: data)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        try configuration.validate()
        let weights = try EmbeddingGemma2WeightStore(
            directory: directory,
            configuration: configuration, base: base, precision: precision, limits: limits)
        let runtime = EmbeddingGemma2Runtime(
            configuration: configuration, tokenizer: tokenizer,
            weights: weights, artifact: artifact)
        try runtime.validateTokenizer()
        try runtime.loadText()
        return EmbeddingGemma2Container(runtime)
    }
}

private final class EmbeddingGemma2Runtime {
    private enum Media {
        case image(EmbeddingGemma2PreparedImage, video: Bool)
        case audio(EmbeddingGemma2PreparedAudio)
        var count: Int {
            switch self {
            case .image(let input, _): input.tokenCount
            case .audio(let input): input.tokenCount
            }
        }
    }

    let configuration: EmbeddingGemma2Configuration
    let tokenizer: EmbeddingGemma2Tokenizer
    let weights: EmbeddingGemma2WeightStore
    let artifact: EmbeddingGemma2Artifact
    private var text: EmbeddingGemma2TextComponent?
    private var vision: EmbeddingGemma2VisionComponent?
    private var audio: EmbeddingGemma2AudioComponent?

    init(
        configuration: EmbeddingGemma2Configuration, tokenizer: EmbeddingGemma2Tokenizer,
        weights: EmbeddingGemma2WeightStore, artifact: EmbeddingGemma2Artifact
    ) {
        self.configuration = configuration
        self.tokenizer = tokenizer
        self.weights = weights
        self.artifact = artifact
    }

    func validateTokenizer() throws {
        let c = configuration
        for id in [
            c.imageToken, c.audioToken, c.videoToken, c.beginImage, c.endImage, c.beginAudio,
            c.endAudio,
        ] {
            guard let token = tokenizer.convertIdToToken(id),
                try tokenizer.encode(text: token, addSpecialTokens: false) == [id]
            else {
                throw EmbeddingGemma2Error.invalidConfiguration(
                    "tokenizer does not preserve modality token \(id)")
            }
        }
    }

    func loadText() throws {
        guard text == nil else { return }
        let component = EmbeddingGemma2TextComponent(configuration)
        try weights.load(component, modality: .text)
        text = component
    }

    private func loadVision() throws {
        guard vision == nil else { return }
        let component = EmbeddingGemma2VisionComponent(configuration)
        try weights.load(component, modality: .vision)
        vision = component
    }

    private func loadAudio() throws {
        guard audio == nil else { return }
        let component = EmbeddingGemma2AudioComponent(configuration)
        try weights.load(component, modality: .audio)
        audio = component
    }

    var loadedModalities: Set<EmbeddingGemma2Modality> {
        var result = Set<EmbeddingGemma2Modality>()
        if text != nil { result.insert(.text) }
        if vision != nil { result.insert(.vision) }
        if audio != nil { result.insert(.audio) }
        return result
    }

    func unload(_ modalities: Set<EmbeddingGemma2Modality>) {
        if modalities.contains(.text) { text = nil }
        if modalities.contains(.vision) { vision = nil }
        if modalities.contains(.audio) { audio = nil }
    }

    private func tokenString(_ id: Int) throws -> String {
        guard let value = tokenizer.convertIdToToken(id) else {
            throw EmbeddingGemma2Error.invalidConfiguration("missing token \(id)")
        }
        return value
    }

    private func checkedText(_ value: String) throws -> String {
        let reserved = Set([
            configuration.imageToken, configuration.audioToken, configuration.videoToken,
            configuration.beginImage, configuration.endImage, configuration.beginAudio,
            configuration.endAudio,
            configuration.text.bosTokenId, configuration.text.eosTokenId,
            configuration.text.padTokenId,
        ])
        guard
            try !tokenizer.encode(text: value, addSpecialTokens: false).contains(
                where: reserved.contains)
        else {
            throw EmbeddingGemma2Error.invalidInput(
                "text contains reserved media or sequence tokens")
        }
        return value
    }

    func embed(
        _ content: [EmbeddingGemma2Content], purpose: EmbeddingGemma2Purpose,
        dimension: EmbeddingGemma2Dimension, imageSoftTokenBudget: Int
    ) async throws -> EmbeddingGemma2Result {
        try Task.checkCancellation()
        guard !content.isEmpty, [70, 140, 280, 560, 1120].contains(imageSoftTokenBudget) else {
            throw EmbeddingGemma2Error.invalidInput(
                "provide content and a supported image soft-token budget")
        }
        let c = configuration
        var prompt: String
        switch purpose {
        case .query(let task): prompt = "task: \(try checkedText(task)) | query: "
        case .document(let title): prompt = "title: \(try checkedText(title)) | text: "
        }
        var media = [Media]()
        var minimumMediaTokens = 0
        var hasContent = false
        func appendImage(_ image: EmbeddingGemma2PreparedImage, video: Bool) throws {
            prompt += try tokenString(c.beginImage)
            prompt += String(
                repeating: try tokenString(video ? c.videoToken : c.imageToken),
                count: image.tokenCount)
            prompt += try tokenString(c.endImage)
            media.append(.image(image, video: video))
            minimumMediaTokens += image.tokenCount + 2
        }
        for item in content {
            try Task.checkCancellation()
            switch item {
            case .text(let string):
                prompt += try checkedText(string)
                hasContent = hasContent || !string.isEmpty
            case .image(let image):
                try appendImage(
                    EmbeddingGemma2MediaProcessor.image(
                        image, softTokenBudget: imageSoftTokenBudget), video: false)
                hasContent = true
            case .video(let video):
                for frame in try await EmbeddingGemma2MediaProcessor.video(video) {
                    try appendImage(frame, video: true)
                }
                hasContent = true
            case .audio(let input):
                let prepared = try await EmbeddingGemma2MediaProcessor.audio(
                    input,
                    maximumTokens: max(0, weights.resources.maximumTokens - minimumMediaTokens - 2))
                prompt += try tokenString(c.beginAudio)
                prompt += String(
                    repeating: try tokenString(c.audioToken), count: prepared.tokenCount)
                prompt += try tokenString(c.endAudio)
                media.append(.audio(prepared))
                minimumMediaTokens += prepared.tokenCount + 2
                hasContent = true
            }
            guard minimumMediaTokens <= weights.resources.maximumTokens else {
                throw EmbeddingGemma2Error.tokenLimit(
                    count: minimumMediaTokens, limit: weights.resources.maximumTokens)
            }
            // Tokenize the whole prefix so interleaved text keeps its exact boundary behavior.
            let count = try tokenizer.encode(text: prompt, addSpecialTokens: true).count
            guard count <= weights.resources.maximumTokens else {
                throw EmbeddingGemma2Error.tokenLimit(
                    count: count, limit: weights.resources.maximumTokens)
            }
        }
        guard hasContent else { throw EmbeddingGemma2Error.invalidInput("empty content") }
        let tokens = try tokenizer.encode(text: prompt, addSpecialTokens: true)
        guard !tokens.isEmpty, tokens.allSatisfy({ (0 ..< c.text.vocabularySize).contains($0) })
        else {
            throw EmbeddingGemma2Error.invalidInput(
                "tokenizer produced an empty sequence or invalid token ID")
        }
        try loadText()
        guard let text else {
            throw EmbeddingGemma2Error.invalidWeights("text encoder unavailable")
        }
        let mediaIds = Set([c.imageToken, c.audioToken, c.videoToken])
        let textIds = tokens.map { mediaIds.contains($0) ? c.text.padTokenId : $0 }
        var embeddings = text.encoder.embedTokens(MLXArray(textIds).expandedDimensions(axis: 0))
        embeddings =
            embeddings * MLXArray(pow(Float(c.text.hiddenSize), 0.5), dtype: embeddings.dtype)
        var featuresById = [Int: [MLXArray]]()
        for item in media {
            try Task.checkCancellation()
            let id: Int
            let features: MLXArray
            switch item {
            case .image(let image, let video):
                try loadVision()
                guard let vision else {
                    throw EmbeddingGemma2Error.invalidWeights("vision encoder unavailable")
                }
                features = try vision(image)
                id = video ? c.videoToken : c.imageToken
            case .audio(let input):
                try loadAudio()
                guard let audio else {
                    throw EmbeddingGemma2Error.invalidWeights("audio encoder unavailable")
                }
                features = try audio(input)
                id = c.audioToken
            }
            guard features.shape == [item.count, c.text.hiddenSize] else {
                throw EmbeddingGemma2Error.featureCount(
                    modality: "media", expected: item.count, actual: features.dim(0))
            }
            try checkedEval(features)
            featuresById[id, default: []].append(features)
        }
        for id in [c.imageToken, c.audioToken, c.videoToken] {
            let indices = tokens.enumerated().compactMap {
                $0.element == id ? Int32($0.offset) : nil
            }
            let parts = featuresById[id] ?? []
            let count = parts.reduce(0) { $0 + $1.dim(0) }
            guard indices.count == count else {
                throw EmbeddingGemma2Error.featureCount(
                    modality: "token \(id)", expected: indices.count, actual: count)
            }
            if !indices.isEmpty {
                embeddings[0, MLXArray(indices), 0...] = concatenated(parts, axis: 0).asType(
                    embeddings.dtype)
            }
        }
        let valid = MLXArray.ones([1, tokens.count], dtype: .bool)
        let hidden = try text.encoder(embeddings, validMask: valid)
        // Prompt, delimiter, and media tokens all participate in the masked mean.
        let mask = valid.asType(.float32).expandedDimensions(axis: -1)
        let pooled = sum(hidden * mask, axis: 1) / sum(mask, axis: 1)
        let truncated = pooled[0, ..<dimension.rawValue].asType(.float32)
        let normalized = truncated / sqrt(sum(truncated.square()))
        try checkedEval(normalized)
        let values = normalized.asArray(Float.self)
        let norm = values.reduce(0.0) { $0 + Double($1) * Double($1) }
        guard values.count == dimension.rawValue, values.allSatisfy(\.isFinite),
            norm.isFinite, abs(norm - 1) <= 1e-4
        else { throw EmbeddingGemma2Error.nonFiniteEmbedding }
        try Task.checkCancellation()
        return EmbeddingGemma2Result(
            values: values, dimension: dimension, artifact: artifact,
            referenceRevision: EmbeddingGemma2Artifact.referenceRevision, tokenCount: tokens.count)
    }
}
