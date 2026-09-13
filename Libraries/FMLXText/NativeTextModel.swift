// Copyright © 2026 Faux Fox.

import Foundation
import MLXLLM
import MLXLMCommon

/// Selects an optional prefill chunk-size policy for native text models.
public enum NativeTextPrefillChunkPolicy: Sendable, Equatable {
    /// Keep the caller's runtime configuration unchanged.
    case configured

    /// Use at least a memory-conscious chunk size for qualified resident MoE models.
    case balanced

    /// Use at least a throughput-oriented chunk size for qualified resident MoE models.
    case throughput
}

/// An audited native model and its matching checkpoint text processor.
/// Keep this value with the runtime so concurrent requests cannot pick another model's tokenizer.
public struct NativeTextModel: Sendable {
    public let text: CheckpointTextProcessor
    public let runtime: ConcurrentTextRuntime
    public let cacheIdentity: PrefixCacheIdentity
    public let toolCallFormat: ToolCallFormat?
    public let reasoningConfig: ReasoningConfig?
    /// The prefill chunk size passed to this model's runtime.
    public let prefillChunkSize: Int
    private let configuration: ConcurrentTextRuntime.Configuration

    public static func load(
        directory: URL, modelRevision: String,
        configuration: ConcurrentTextRuntime.Configuration,
        extraEOSTokens: Set<String> = [],
        mtpCompanionDirectory: URL? = nil,
        loadPolicy: NativeTextModelLoadPolicy = .resident,
        prefillChunkPolicy: NativeTextPrefillChunkPolicy = .balanced
    ) async throws -> Self {
        let text = try await CheckpointTextProcessor.load(
            directory: directory, extraEOSTokens: extraEOSTokens)
        let model = try await NativeTextModelLoader.load(
            directory: directory, policy: loadPolicy)
        let drafter: Qwen35MTPDraftModel?
        if case .streamedExperts = loadPolicy {
            guard mtpCompanionDirectory == nil else {
                throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                    "Streaming experts does not support an MTP companion")
            }
            drafter = nil
        } else if case .edge0 = loadPolicy {
            guard mtpCompanionDirectory == nil else {
                throw NativeTextModelLoadingError.streamedExpertsUnsupported(
                    "Edge0 does not support an MTP companion")
            }
            drafter = nil
        } else if let mtpCompanionDirectory {
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
        let baseIdentity = try text.cacheIdentity(
            modelRevision: modelRevision, cacheLayoutRevision: "fmlx-text-v1/\(quantization)")
        let identity: PrefixCacheIdentity
        if case .edge0 = loadPolicy {
            identity = PrefixCacheIdentity(
                modelRevision: baseIdentity.modelRevision,
                tokenizerRevision: baseIdentity.tokenizerRevision,
                chatTemplateRevision: baseIdentity.chatTemplateRevision,
                adapterRevision: "edge0-recover-lora-dbdef1af692986ad1937562c0d2aab7f",
                cacheLayoutRevision: baseIdentity.cacheLayoutRevision + "/edge0-ae1ee2d")
        } else {
            identity = baseIdentity
        }
        let toolCallFormat = ToolCallFormat.resolved(
            forTokenizerDirectory: directory, modelFormat: model.toolCallFormat)
        let reasoningConfig = model.reasoningConfig
        let prefillChunkSize = resolvedPrefillChunkSize(
            configured: configuration.prefillChunkSize, policy: prefillChunkPolicy,
            isResidentQwen35MoE: loadPolicy == .resident && model is Qwen35MoEModel)
        let runtimeConfiguration = configuration.replacingPrefillChunkSize(prefillChunkSize)
        let runtime = try ConcurrentTextRuntime(
            model: model, identity: identity, configuration: runtimeConfiguration, drafter: drafter)
        return Self(
            text: text, runtime: runtime, cacheIdentity: identity,
            toolCallFormat: toolCallFormat, reasoningConfig: reasoningConfig,
            prefillChunkSize: prefillChunkSize, configuration: runtimeConfiguration)
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

func resolvedPrefillChunkSize(
    configured: Int, policy: NativeTextPrefillChunkPolicy, isResidentQwen35MoE: Bool
) -> Int {
    guard isResidentQwen35MoE else { return configured }
    return switch policy {
    case .configured:
        configured
    case .balanced:
        max(configured, 256)
    case .throughput:
        max(configured, 512)
    }
}

extension ConcurrentTextRuntime.Configuration {
    fileprivate func replacingPrefillChunkSize(_ prefillChunkSize: Int) -> Self {
        .init(
            memoryBudgetBytes: memoryBudgetBytes,
            prefixCacheBytes: prefixCacheBytes,
            workingMemoryBytes: workingMemoryBytes,
            maxActiveRequests: maxActiveRequests,
            maxQueuedRequests: maxQueuedRequests,
            maxPromptTokens: maxPromptTokens,
            maxOutputTokens: maxOutputTokens,
            prefillChunkSize: prefillChunkSize,
            streamBufferSize: streamBufferSize,
            batchDecode: batchDecode,
            interactiveReservedSlots: interactiveReservedSlots,
            cacheQuantization: cacheQuantization,
            persistentCache: persistentCache,
            speculativeAdaptation: speculativeAdaptation,
            speculativeBlockSize: speculativeBlockSize)
    }
}
