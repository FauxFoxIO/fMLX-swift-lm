// Copyright © 2026 Faux Fox.

import Foundation
import MLXLLM
import MLXLMCommon

/// An audited native model and its matching checkpoint text processor.
/// Keep this value with the runtime so concurrent requests cannot pick another model's tokenizer.
public struct NativeTextModel: Sendable {
    public let text: CheckpointTextProcessor
    public let runtime: ConcurrentTextRuntime
    public let cacheIdentity: PrefixCacheIdentity
    public let toolCallFormat: ToolCallFormat?
    public let reasoningConfig: ReasoningConfig?
    private let configuration: ConcurrentTextRuntime.Configuration

    public static func load(
        directory: URL, modelRevision: String,
        configuration: ConcurrentTextRuntime.Configuration,
        extraEOSTokens: Set<String> = [],
        mtpCompanionDirectory: URL? = nil
    ) async throws -> Self {
        let text = try await CheckpointTextProcessor.load(
            directory: directory, extraEOSTokens: extraEOSTokens)
        let model = try await NativeTextModelLoader.load(directory: directory)
        let drafter: Qwen35MTPDraftModel?
        if let mtpCompanionDirectory {
            drafter = try await NativeTextModelLoader.loadMTP(directory: mtpCompanionDirectory)
        } else {
            drafter = try await NativeTextModelLoader.loadCombinedMTP(directory: directory)
        }
        try Task.checkCancellation()
        try text.verifyAssets(directory: directory)
        guard model.vocabularySize == text.vocabularySize else {
            throw CheckpointTextError.invalidConfiguration(
                "Loaded model and tokenizer vocabulary dimensions differ")
        }
        let quantization =
            configuration.cacheQuantization.map { "kv\($0.bits)-group\($0.groupSize)" } ?? "native"
        let identity = try text.cacheIdentity(
            modelRevision: modelRevision, cacheLayoutRevision: "fmlx-text-v1/\(quantization)")
        let toolCallFormat = ToolCallFormat.resolved(
            forTokenizerDirectory: directory, modelFormat: model.toolCallFormat)
        let reasoningConfig = model.reasoningConfig
        let runtime = try ConcurrentTextRuntime(
            model: model, identity: identity, configuration: configuration, drafter: drafter)
        return Self(
            text: text, runtime: runtime, cacheIdentity: identity,
            toolCallFormat: toolCallFormat, reasoningConfig: reasoningConfig,
            configuration: configuration)
    }

    /// Constructs a raw runtime request using this model's exact chat tokens and stop IDs.
    /// String-stop checkpoints need a text-stream stop matcher and are explicitly rejected here.
    public func prepareRequest(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]? = nil,
        additionalContext: [String: any Sendable]? = nil,
        maximumOutputTokens: Int = 512, temperature: Float? = nil,
        topP: Float? = nil, topK: Int? = nil, seed: UInt64? = nil,
        priority: ConcurrentTextRuntime.Priority = .interactive, prefixTokenCount: Int = 0,
        cachePromptPrefix: Bool = false
    ) throws -> ConcurrentTextRuntime.Request {
        guard text.stopStrings.isEmpty else {
            throw CheckpointTextError.stringStopsRequireTextGeneration
        }
        guard maximumOutputTokens <= configuration.maxOutputTokens else {
            throw ConcurrentTextRuntimeError.invalidRequest
        }
        let tokens = try text.prepareChat(
            messages: messages, tools: tools, additionalContext: additionalContext)
        let resolvedPrefixTokenCount = cachePromptPrefix ? tokens.count - 1 : prefixTokenCount
        guard tokens.count <= configuration.maxPromptTokens, resolvedPrefixTokenCount >= 0,
            resolvedPrefixTokenCount < tokens.count
        else {
            throw ConcurrentTextRuntimeError.invalidRequest
        }
        try text.validateContext(
            promptTokenCount: tokens.count, maximumOutputTokens: maximumOutputTokens)
        return ConcurrentTextRuntime.Request(
            tokens: tokens, maxTokens: maximumOutputTokens,
            temperature: temperature ?? text.temperature,
            topP: topP ?? text.topP, topK: topK ?? text.topK,
            seed: seed, stopTokenIDs: text.stopTokenIDs,
            priority: priority,
            prefixTokenCount: resolvedPrefixTokenCount, cacheIdentity: cacheIdentity)
    }
}
