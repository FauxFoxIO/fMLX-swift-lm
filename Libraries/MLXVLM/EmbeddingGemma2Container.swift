// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Standalone, serialized native embeddings. No causal model, chat session, or generation cache.
public final class EmbeddingGemma2Container: Sendable {
    public let artifact: EmbeddingGemma2Artifact
    public let resources: EmbeddingGemma2Resources
    private let state: SerialAccessContainer<EmbeddingGemma2Runtime>

    fileprivate init(_ runtime: consuming EmbeddingGemma2Runtime) {
        artifact = runtime.artifact
        resources = runtime.resources
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
    /// Releasing `.text` releases every modality.
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
        return try await loadContainer(
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
        return try await loadContainer(
            from: directory, tokenizer: EmbeddingGemma2Tokenizer(tokenizer),
            artifact: artifact, precision: precision, limits: limits)
    }

    private static func loadContainer(
        from directory: URL, tokenizer: EmbeddingGemma2Tokenizer,
        artifact: EmbeddingGemma2Artifact, precision: EmbeddingGemma2Precision,
        limits: EmbeddingGemma2ResourceLimits
    ) async throws -> EmbeddingGemma2Container {
        guard directory.isFileURL, !artifact.identifier.isEmpty, !artifact.revision.isEmpty,
            artifact.revision != "main", artifact.revision != "latest"
        else {
            throw EmbeddingGemma2Error.invalidConfiguration(
                "provide a local snapshot and immutable artifact revision")
        }
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try JSONDecoder().decode(EmbeddingGemma2Configuration.self, from: data)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
        let inventory = try EmbeddingGemma2CheckpointInventory(directory: directory, limits: limits)
        let runtime = EmbeddingGemma2Runtime(
            directory: directory, configuration: configuration,
            perLayerQuantization: base.perLayerQuantization, tokenizer: tokenizer,
            inventory: inventory, artifact: artifact, precision: precision)
        try runtime.validateTokenizer()
        _ = try await runtime.loadModel()
        return EmbeddingGemma2Container(runtime)
    }
}

/// Drives the shared `EmbeddingGemma2` encoder and its reference processors.
private final class EmbeddingGemma2Runtime {
    /// A media item ready to encode, and the soft tokens it will produce.
    private enum Media {
        case image(MLXArray)
        case video(MLXArray)
        case audio(MLXArray, tokens: Int)
    }

    private enum Part {
        case text([Int])
        case media(Media)
    }

    let directory: URL
    let configuration: EmbeddingGemma2Configuration
    let perLayerQuantization: BaseConfiguration.PerLayerQuantization?
    let tokenizer: EmbeddingGemma2Tokenizer
    let inventory: EmbeddingGemma2CheckpointInventory
    let artifact: EmbeddingGemma2Artifact
    let precision: EmbeddingGemma2Precision
    private var model: EmbeddingGemma2?
    /// Media encoders load with the text model when the checkpoint dtype needs a cast.
    private var loadsMediaWithText = false
    private var imageProcessor: EmbeddingGemma2ImageProcessor?
    private var videoProcessor: EmbeddingGemma2VideoProcessor?
    private var audioProcessor: EmbeddingGemma2AudioProcessor?

    var resources: EmbeddingGemma2Resources { inventory.resources }

    init(
        directory: URL, configuration: EmbeddingGemma2Configuration,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization?,
        tokenizer: EmbeddingGemma2Tokenizer, inventory: EmbeddingGemma2CheckpointInventory,
        artifact: EmbeddingGemma2Artifact, precision: EmbeddingGemma2Precision
    ) {
        self.directory = directory
        self.configuration = configuration
        self.perLayerQuantization = perLayerQuantization
        self.tokenizer = tokenizer
        self.inventory = inventory
        self.artifact = artifact
        self.precision = precision
    }

    private var markerTokenIDs: [Int] {
        var ids = configuration.softTokenIDs
        if let vision = configuration.vision {
            ids += [vision.beginImageTokenID, vision.endImageTokenID]
        }
        if let audio = configuration.audio {
            ids += [audio.beginAudioTokenID, audio.endAudioTokenID]
        }
        return ids
    }

    func validateTokenizer() throws {
        _ = try tokenizer.sequenceBoundaries()
        for id in markerTokenIDs {
            guard let token = tokenizer.convertIdToToken(id),
                try tokenizer.encode(text: token, addSpecialTokens: false) == [id]
            else {
                throw EmbeddingGemma2Error.invalidConfiguration(
                    "tokenizer does not preserve modality token \(id)")
            }
        }
    }

    func loadModel() async throws -> EmbeddingGemma2 {
        if let model { return model }
        try inventory.verifyFiles()
        let model = EmbeddingGemma2(configuration)
        let casts = !inventory.stores(precision)
        try await loadWeights(
            modelDirectory: directory, model: model, perLayerQuantization: perLayerQuantization,
            componentLoading: casts ? .immediate : .onFirstUse)
        if casts {
            let parameters = model.parameters().flattened().map { name, value in
                (name, value.dtype == .uint32 ? value : value.asType(precision.dtype))
            }
            try model.update(parameters: ModuleParameters.unflattened(parameters), verify: .all)
            try checkedEval(model)
        }
        model.train(false)
        loadsMediaWithText = casts
        self.model = model
        return model
    }

    var loadedModalities: Set<EmbeddingGemma2Modality> {
        guard let model else { return [] }
        var result: Set<EmbeddingGemma2Modality> = [.text]
        if model.loadedComponents.contains(.vision) { result.insert(.vision) }
        if model.loadedComponents.contains(.audio) { result.insert(.audio) }
        return result
    }

    /// Unloading text releases the whole model.
    func unload(_ modalities: Set<EmbeddingGemma2Modality>) {
        guard let model else { return }
        var components = Set<ModelComponent>()
        if modalities.contains(.vision) { components.insert(.vision) }
        if modalities.contains(.audio) { components.insert(.audio) }
        if modalities.contains(.text)
            || (loadsMediaWithText && !components.isDisjoint(with: model.loadedComponents))
        {
            self.model = nil
        } else {
            model.unloadMediaComponents(components)
        }
    }

    private func checkedText(_ value: String) throws -> String {
        let (begin, end) = try tokenizer.sequenceBoundaries()
        var reserved = Set(markerTokenIDs + [begin, end])
        if let pad = tokenizer.convertTokenToId("<pad>") { reserved.insert(pad) }
        guard
            try !tokenizer.encode(text: value, addSpecialTokens: false).contains(
                where: reserved.contains)
        else {
            throw EmbeddingGemma2Error.invalidInput(
                "text contains reserved media or sequence tokens")
        }
        return value
    }

    private func images() throws -> EmbeddingGemma2ImageProcessor {
        guard configuration.vision != nil else {
            throw EmbeddingGemma2Error.invalidInput("this checkpoint cannot embed images")
        }
        let processor = try imageProcessor ?? EmbeddingGemma2ImageProcessor(directory: directory)
        imageProcessor = processor
        return processor
    }

    private func videos() throws -> EmbeddingGemma2VideoProcessor {
        guard configuration.vision?.videoTokenID != nil else {
            throw EmbeddingGemma2Error.invalidInput("this checkpoint cannot embed video")
        }
        let processor = try videoProcessor ?? EmbeddingGemma2VideoProcessor(directory: directory)
        videoProcessor = processor
        return processor
    }

    private func audio() throws -> EmbeddingGemma2AudioProcessor {
        guard configuration.audio != nil else {
            throw EmbeddingGemma2Error.invalidInput("this checkpoint cannot embed audio")
        }
        let processor = try audioProcessor ?? EmbeddingGemma2AudioProcessor(directory: directory)
        guard processor.sampleRate == EmbeddingGemma2Waveform.sampleRate else {
            throw EmbeddingGemma2Error.invalidConfiguration("unsupported audio sample rate")
        }
        audioProcessor = processor
        return processor
    }

    /// Soft tokens of an image or video frame of these pixel dimensions.
    private func imageTokens(_ pixels: MLXArray) throws -> Int {
        try images().configuration.softTokenCount(height: pixels.dim(2), width: pixels.dim(3))
    }

    /// The audio tower subsamples its frames twice by two.
    private static func audioTokens(frames: Int) -> Int {
        let first = (frames + 1) / 2
        return (first + 1) / 2
    }

    /// Sequence length, including `<bos>`, `<eos>` and each media block's markers.
    private func length(_ parts: [Part]) throws -> Int {
        try parts.reduce(2) { total, part in
            switch part {
            case .text(let tokens): return total + tokens.count
            case .media(.image(let pixels)): return total + (try imageTokens(pixels)) + 2
            case .media(.video(let pixels)):
                return total + pixels.dim(0) * ((try imageTokens(pixels)) + 2)
            case .media(.audio(_, let tokens)): return total + tokens + 2
            }
        }
    }

    func embed(
        _ content: [EmbeddingGemma2Content], purpose: EmbeddingGemma2Purpose,
        dimension: EmbeddingGemma2Dimension, imageSoftTokenBudget: Int
    ) async throws -> EmbeddingGemma2Result {
        try Task.checkCancellation()
        guard !content.isEmpty,
            EmbeddingGemma2MediaProcessor.softTokenBudgets.contains(imageSoftTokenBudget)
        else {
            throw EmbeddingGemma2Error.invalidInput(
                "provide content and a supported image soft-token budget")
        }
        guard dimension.rawValue <= configuration.embeddingDim else {
            throw EmbeddingGemma2Error.invalidConfiguration(
                "dimension exceeds the checkpoint embedding size")
        }
        let limit = resources.maximumTokens
        var text: String
        switch purpose {
        case .query(let task): text = "task: \(try checkedText(task)) | query: "
        case .document(let title): text = "title: \(try checkedText(title)) | text: "
        }

        // Prepare every item and check the sequence length before any encoder runs.
        var parts = [Part]()
        var hasContent = false
        func flushText() throws {
            guard !text.isEmpty else { return }
            parts.append(.text(try tokenizer.encode(text: text, addSpecialTokens: false)))
            text = ""
        }
        for item in content {
            try Task.checkCancellation()
            switch item {
            case .text(let string):
                text += try checkedText(string)
                hasContent = hasContent || !string.isEmpty
                continue
            case .image(let image):
                try flushText()
                parts.append(
                    .media(
                        .image(
                            try EmbeddingGemma2MediaProcessor.image(
                                image, processor: try images(), budget: imageSoftTokenBudget))))
            case .video(let video):
                try flushText()
                parts.append(
                    .media(
                        .video(
                            try await EmbeddingGemma2MediaProcessor.video(
                                video, images: try images(), video: try videos()))))
            case .audio(let input):
                try flushText()
                let processor = try audio()
                let samples = try await EmbeddingGemma2Waveform.samples(input)
                let features = processor.features(samples: MLXArray(samples))
                try checkedEval(features)
                guard features.dim(0) > 0 else {
                    throw EmbeddingGemma2Error.invalidInput(
                        "Audio requires at least one feature frame")
                }
                parts.append(
                    .media(.audio(features, tokens: Self.audioTokens(frames: features.dim(0)))))
            }
            hasContent = true
            let count = try length(parts)
            guard count <= limit else {
                throw EmbeddingGemma2Error.tokenLimit(count: count, limit: limit)
            }
        }
        try flushText()
        guard hasContent else { throw EmbeddingGemma2Error.invalidInput("empty content") }
        let expected = try length(parts)
        guard expected <= limit else {
            throw EmbeddingGemma2Error.tokenLimit(count: expected, limit: limit)
        }

        // Encode one media item at a time so only one encoder's activations are alive.
        let model = try await loadModel()
        var segments = [EmbeddingGemma2Sequence.Segment]()
        var softTokens = [MLXArray]()
        for part in parts {
            try Task.checkCancellation()
            let features: MLXArray?
            switch part {
            case .text(let tokens):
                segments.append(.text(tokens))
                continue
            case .media(.image(let pixels)), .media(.video(let pixels)):
                try await model.loadMediaComponents([.vision])
                features = try model.imageFeatures(pixels)
            case .media(.audio(let frames, _)):
                try await model.loadMediaComponents([.audio])
                features = try model.audioFeatures(frames)
            }
            guard let features else {
                throw EmbeddingGemma2Error.invalidWeights("media encoder unavailable")
            }
            try checkedEval(features)
            switch part {
            case .media(.image): segments.append(.image(tokens: features.dim(1)))
            case .media(.video):
                segments.append(.video(frames: features.dim(0), tokensPerFrame: features.dim(1)))
            default: segments.append(.audio(tokens: features.dim(1)))
            }
            softTokens.append(features.reshaped(-1, features.dim(-1)))
        }

        let (begin, end) = try tokenizer.sequenceBoundaries()
        let tokens = try EmbeddingGemma2Sequence.tokens(
            segments, configuration: configuration, beginOfSequence: begin,
            endOfSequence: end, limit: limit)
        let textTokens = segments.reduce(0) { total, segment in
            if case .text(let tokens) = segment { return total + tokens.count }
            return total
        }
        let mediaTokens = softTokens.reduce(0) { $0 + $1.dim(0) }
        let blocks = segments.reduce(0) { total, segment in
            switch segment {
            case .text: return total
            case .video(let frames, _): return total + 2 * frames
            case .image, .audio: return total + 2
            }
        }
        // The sequence builder trims text to fit; embedding input must never change silently.
        guard tokens.count == 2 + textTokens + mediaTokens + blocks else {
            throw EmbeddingGemma2Error.tokenLimit(
                count: 2 + textTokens + mediaTokens + blocks, limit: limit)
        }
        guard tokens.allSatisfy({ (0 ..< configuration.vocabularySize).contains($0) }) else {
            throw EmbeddingGemma2Error.invalidInput("tokenizer produced an invalid token ID")
        }

        let vector = model.embed(
            inputIds: MLXArray(tokens.map(Int32.init), [1, tokens.count]), attentionMask: nil,
            softTokens: softTokens.isEmpty ? nil : concatenated(softTokens, axis: 0))
        try checkedEval(vector)
        try Task.checkCancellation()
        let truncated = vector.asArray(Float.self).prefix(dimension.rawValue)
        let norm = truncated.reduce(0.0) { $0 + Double($1) * Double($1) }.squareRoot()
        guard truncated.count == dimension.rawValue, truncated.allSatisfy(\.isFinite),
            norm.isFinite, norm > 0
        else { throw EmbeddingGemma2Error.nonFiniteEmbedding }
        let values = truncated.map { Float(Double($0) / norm) }
        return EmbeddingGemma2Result(
            values: values, dimension: dimension, artifact: artifact,
            referenceRevision: EmbeddingGemma2Artifact.referenceRevision, tokenCount: tokens.count)
    }
}
