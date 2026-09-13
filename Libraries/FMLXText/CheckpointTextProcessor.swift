// Copyright © 2026 Faux Fox.

import CryptoKit
import Foundation
import MLXLMCommon
import Tokenizers

public enum CheckpointTextError: LocalizedError, Equatable, Sendable {
    case invalidConfiguration(String)
    case assetsChangedDuringLoad
    case invalidTokenID(Int)
    case contextExceeded(promptTokens: Int, outputTokens: Int, capacity: Int)
    case stringStopsRequireTextGeneration

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let reason): reason
        case .assetsChangedDuringLoad:
            "Checkpoint text assets changed during loading; reload the model."
        case .invalidTokenID(let id): "Token \(id) is outside this checkpoint's vocabulary."
        case .contextExceeded(let prompt, let output, let capacity):
            "The formatted prompt (\(prompt) tokens) and output allowance (\(output)) exceed this model's \(capacity)-token context."
        case .stringStopsRequireTextGeneration:
            "This checkpoint requires string-stop matching, which is unavailable in the raw-token request path."
        }
    }
}

/// Immutable, local-only text processing loaded from the same checkpoint as the model.
/// The tokenizer engine is shared code; vocabulary, normalization, special tokens and
/// chat templates always come from this checkpoint, never another model's defaults.
public struct CheckpointTextProcessor: Sendable {
    private let tokenizer: any Tokenizers.Tokenizer
    public let vocabularySize: Int
    public let contextWindowTokens: Int?
    public let stopTokenIDs: Set<Int>
    public let stopStrings: Set<String>
    public let tokenizerRevision: String
    public let chatTemplateRevision: String
    public let temperature: Float
    public let topP: Float
    public let topK: Int
    public var hasChatTemplate: Bool { tokenizer.hasChatTemplate }

    func verifyAssets(directory: URL) throws {
        guard
            try TextAssets(directory: directory).digest(names: TextAssets.names)
                == tokenizerRevision
        else {
            throw CheckpointTextError.assetsChangedDuringLoad
        }
    }

    public static func load(directory: URL, extraEOSTokens: Set<String> = []) async throws -> Self {
        guard directory.isFileURL else {
            throw CheckpointTextError.invalidConfiguration(
                "A local checkpoint directory is required")
        }
        let assets = try TextAssets(directory: directory)
        let decoder = JSONDecoder()
        let base = try decoder.decode(BaseConfiguration.self, from: assets.required("config.json"))
        let metadata = try decoder.decode(Metadata.self, from: assets.required("config.json"))
        let vocabularySize = metadata.text?.vocabularySize ?? metadata.vocabularySize
        guard let vocabularySize, vocabularySize > 0 else {
            throw CheckpointTextError.invalidConfiguration(
                "config.json must declare the text vocabulary size")
        }
        let window = metadata.text?.contextWindow ?? metadata.contextWindow
        if let window, window <= 0 {
            throw CheckpointTextError.invalidConfiguration(
                "The model context window must be positive")
        }
        let raw = try JSONSerialization.jsonObject(with: assets.required("tokenizer.json"))
        guard let document = raw as? [String: Any] else {
            throw CheckpointTextError.invalidConfiguration("tokenizer.json must be an object")
        }
        for key in ["truncation", "padding"] {
            if let value = document[key], !(value is NSNull) {
                throw CheckpointTextError.invalidConfiguration(
                    "Inference requires tokenizer \(key) to be disabled")
            }
        }
        let tokenizer = try await AutoTokenizer.from(directory: directory)
        try Task.checkCancellation()
        guard try TextAssets(directory: directory) == assets else {
            throw CheckpointTextError.assetsChangedDuringLoad
        }
        guard tokenizer.getVocabSize(withAddedTokens: true) <= vocabularySize else {
            throw CheckpointTextError.invalidConfiguration(
                "Tokenizer vocabulary exceeds model embeddings")
        }
        let generation = try assets.files["generation_config.json"].map {
            try decoder.decode(GenerationConfigFile.self, from: $0)
        }
        let temperature =
            generation?.doSample == false
            ? 0 : generation?.temperature ?? 0.6
        let topP = generation?.topP ?? 1
        let topK = generation?.topK ?? 0
        guard temperature.isFinite, temperature >= 0,
            topP.isFinite, topP > 0, topP <= 1,
            topK >= 0
        else {
            throw CheckpointTextError.invalidConfiguration(
                "generation_config.json contains invalid sampling parameters"
            )
        }
        var stops = generation?.eosTokenIds.map { Set($0.values) } ?? base.effectiveEOSTokenIds
        for special in [tokenizer.bosToken, tokenizer.eosToken, tokenizer.unknownToken].compactMap({
            $0
        }) {
            guard tokenizer.convertTokenToId(special) != nil else {
                throw CheckpointTextError.invalidConfiguration(
                    "Special token is absent from the checkpoint vocabulary: \(special)")
            }
        }
        if let eos = tokenizer.eosToken, let id = tokenizer.convertTokenToId(eos) {
            stops.insert(id)
        }
        for token in extraEOSTokens {
            guard let id = tokenizer.convertTokenToId(token) else {
                throw CheckpointTextError.invalidConfiguration(
                    "Extra EOS token is absent from the checkpoint vocabulary: \(token)")
            }
            stops.insert(id)
        }
        for id in stops {
            guard (0 ..< vocabularySize).contains(id), tokenizer.convertIdToToken(id) != nil else {
                throw CheckpointTextError.invalidTokenID(id)
            }
        }
        return Self(
            tokenizer: tokenizer, vocabularySize: vocabularySize, contextWindowTokens: window,
            stopTokenIDs: stops, stopStrings: generation?.stopStrings ?? [],
            tokenizerRevision: assets.digest(names: TextAssets.names),
            chatTemplateRevision: assets.digest(names: TextAssets.templateNames),
            temperature: temperature, topP: topP, topK: topK)
    }

    public func encode(_ text: String, addSpecialTokens: Bool = true) throws -> [Int] {
        let tokens = try tokenizer.encode(
            text: text, textPair: nil, addSpecialTokens: addSpecialTokens)
        try validate(tokens)
        return tokens
    }

    /// Applies the checkpoint's default or tool-use template, with no truncation.
    /// Preserve structured tool-call/result fields in the supplied message dictionaries.
    public func prepareChat(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]? = nil,
        addGenerationPrompt: Bool = true, templateName: String? = nil,
        additionalContext: [String: any Sendable]? = nil
    ) throws -> [Int] {
        // Template options may control model features (for example enable_thinking),
        // but must not replace messages or special tokens behind the caller's back.
        let reserved: Set<String> = [
            "messages", "tools", "add_generation_prompt", "bos_token", "eos_token", "unk_token",
            "pad_token", "sep_token", "cls_token", "mask_token", "additional_special_tokens",
        ]
        if let key = additionalContext?.keys.first(where: { reserved.contains($0) }) {
            throw CheckpointTextError.invalidConfiguration("Reserved chat-template option: \(key)")
        }
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages, chatTemplate: templateName.map { .name($0) },
            addGenerationPrompt: addGenerationPrompt, truncation: false, maxLength: nil,
            tools: tools, additionalContext: additionalContext)
        try validate(tokens)
        return tokens
    }

    public func decode(_ tokens: [Int], skipSpecialTokens: Bool = false) throws -> String {
        try validate(tokens)
        return try tokenizer.decode(tokenIds: tokens, skipSpecialTokens: skipSpecialTokens)
    }

    public func validateContext(
        promptTokenCount: Int, maximumOutputTokens: Int, capacity: Int? = nil
    ) throws {
        guard promptTokenCount > 0, maximumOutputTokens > 0 else {
            throw CheckpointTextError.invalidConfiguration(
                "Prompt and output token counts must be positive")
        }
        let limits = [capacity, contextWindowTokens].compactMap { $0 }
        if let limit = limits.min(),
            maximumOutputTokens > limit || promptTokenCount > limit - maximumOutputTokens
        {
            throw CheckpointTextError.contextExceeded(
                promptTokens: promptTokenCount, outputTokens: maximumOutputTokens, capacity: limit)
        }
    }

    /// Caller supplies immutable weight/layout revisions. Tokenizer/template revisions
    /// are hashes of the checkpoint assets, not paths, model names or branch names.
    public func cacheIdentity(modelRevision: String, cacheLayoutRevision: String) throws
        -> PrefixCacheIdentity
    {
        guard !modelRevision.isEmpty, !cacheLayoutRevision.isEmpty else {
            throw CheckpointTextError.invalidConfiguration(
                "Immutable model and cache-layout revisions are required")
        }
        return PrefixCacheIdentity(
            modelRevision: modelRevision, tokenizerRevision: tokenizerRevision,
            chatTemplateRevision: chatTemplateRevision, adapterRevision: "none",
            cacheLayoutRevision: cacheLayoutRevision)
    }

    /// Each generation owns its own incremental decoder; mutable decoder state is never shared.
    public func makeDecoder(skipSpecialTokens: Bool = false) -> CheckpointTextDecoder {
        CheckpointTextDecoder(
            processor: self,
            decoder: tokenizer.streamingDetokenizer(skipSpecialTokens: skipSpecialTokens))
    }

    fileprivate func validate(_ tokens: [Int]) throws {
        for token in tokens
        where !(0 ..< vocabularySize).contains(token) || tokenizer.convertIdToToken(token) == nil {
            throw CheckpointTextError.invalidTokenID(token)
        }
    }

    private struct Metadata: Decodable {
        let vocabularySize: Int?
        let contextWindow: Int?
        let text: TextMetadata?
        enum CodingKeys: String, CodingKey {
            case vocabularySize = "vocab_size"
            case contextWindow = "max_position_embeddings"
            case text = "text_config"
        }
    }
    private struct TextMetadata: Decodable {
        let vocabularySize: Int?
        let contextWindow: Int?
        enum CodingKeys: String, CodingKey {
            case vocabularySize = "vocab_size"
            case contextWindow = "max_position_embeddings"
        }
    }
}

/// A single-consumer decoder with the owning checkpoint's vocabulary validation.
public final class CheckpointTextDecoder {
    private let processor: CheckpointTextProcessor
    private let decoder: Tokenizers.StreamingDetokenizer
    fileprivate init(processor: CheckpointTextProcessor, decoder: Tokenizers.StreamingDetokenizer) {
        self.processor = processor
        self.decoder = decoder
    }
    public func consume(_ token: Int) throws -> String? {
        try processor.validate([token])
        return try decoder.consume(token)
    }
}

private struct TextAssets: Equatable {
    static let templateNames = [
        "tokenizer_config.json", "chat_template.jinja", "chat_template.json",
    ]
    static let names =
        [
            "config.json", "generation_config.json", "tokenizer.json", "special_tokens_map.json",
            "added_tokens.json",
        ] + templateNames
    let files: [String: Data]

    init(directory: URL) throws {
        var files: [String: Data] = [:]
        for name in Self.names {
            let url = directory.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) {
                let data = try Data(contentsOf: url)
                if name.hasSuffix(".json") {
                    guard
                        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                    else {
                        throw CheckpointTextError.invalidConfiguration(
                            "\(name) must contain a JSON object")
                    }
                    if let templates = object["chat_template"] as? [[String: Any]] {
                        let names = templates.compactMap { $0["name"] as? String }
                        guard names.count == templates.count, Set(names).count == names.count else {
                            throw CheckpointTextError.invalidConfiguration(
                                "Chat-template names must be present and unique")
                        }
                    }
                }
                files[name] = data
            }
        }
        self.files = files
        _ = try required("config.json")
        _ = try required("tokenizer.json")
    }
    func required(_ name: String) throws -> Data {
        guard let data = files[name] else {
            throw CheckpointTextError.invalidConfiguration("Missing checkpoint file: \(name)")
        }
        return data
    }
    func digest(names: [String]) -> String {
        var hash = SHA256()
        hash.update(data: Data("fmlx-text/tokenizers-0.7.3/v1".utf8))
        for name in names.sorted() {
            hash.update(data: Data("\n\(name):\(files[name]?.count ?? -1):".utf8))
            if let data = files[name] { hash.update(data: data) }
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
