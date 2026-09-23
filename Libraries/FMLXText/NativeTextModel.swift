// Copyright © 2026 Faux Fox.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

/// Conservative pre-load admission result for a resident native model.
public struct NativeTextModelMemoryAdmission: Sendable, Equatable {
    public let checkpointBytes: Int
    public let requiredBytes: Int
    public let deviceBudgetBytes: Int
    public var isAdmitted: Bool { requiredBytes <= deviceBudgetBytes }
}

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
        mtpRevision: String? = nil,
        transformContract: String? = nil,
        loadPolicy: NativeTextModelLoadPolicy = .resident,
        prefillChunkPolicy: NativeTextPrefillChunkPolicy = .balanced
    ) async throws -> Self {
        if mtpCompanionDirectory != nil, mtpRevision?.isEmpty != false {
            throw CheckpointTextError.invalidConfiguration(
                "An immutable MTP revision is required for a companion checkpoint")
        }
        let admission = try memoryAdmission(
            directory: directory, mtpCompanionDirectory: mtpCompanionDirectory,
            configuration: configuration)
        guard admission.isAdmitted else {
            throw ConcurrentTextRuntimeError.memoryBudgetExceeded
        }
        let text = try await CheckpointTextProcessor.load(
            directory: directory, extraEOSTokens: extraEOSTokens)
        let model = try await NativeTextModelLoader.load(
            directory: directory, policy: loadPolicy)
        let drafter: (any IncrementalMTPDrafterModel)?
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
            drafter = try await NativeTextModelLoader.loadSpeculativeDrafter(
                directory: mtpCompanionDirectory)
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
        let draftRevision = drafter == nil ? "none" : mtpRevision ?? "\(modelRevision)/combined-mtp"
        let resolvedTransformContract =
            transformContract
            ?? (model as? any InferenceArtifactIdentityProviding)?.transformContractRevision
            ?? "none"
        let baseIdentity = try text.cacheIdentity(
            modelRevision: modelRevision, cacheLayoutRevision: "fmlx-text-v1/\(quantization)",
            draftRevision: draftRevision, quantizationContract: quantization,
            transformContract: resolvedTransformContract)
        let identity: PrefixCacheIdentity
        if case .edge0 = loadPolicy {
            identity = PrefixCacheIdentity(
                modelRevision: baseIdentity.modelRevision,
                tokenizerRevision: baseIdentity.tokenizerRevision,
                chatTemplateRevision: baseIdentity.chatTemplateRevision,
                adapterRevision: "edge0-recover-lora-dbdef1af692986ad1937562c0d2aab7f",
                cacheLayoutRevision: baseIdentity.cacheLayoutRevision + "/edge0-ae1ee2d",
                draftRevision: baseIdentity.draftRevision,
                quantizationContract: baseIdentity.quantizationContract,
                transformContract: baseIdentity.transformContract)
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

    /// Estimates resident checkpoint and runtime storage before any tensors are loaded.
    /// The device ceiling is deliberately conservative so oversized models fail before
    /// unified-memory pressure can terminate an iPhone process.
    public static func memoryAdmission(
        directory: URL, mtpCompanionDirectory: URL? = nil,
        configuration: ConcurrentTextRuntime.Configuration
    ) throws -> NativeTextModelMemoryAdmission {
        func checkpointBytes(_ directory: URL) throws -> Int {
            let urls = try FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.fileSizeKey],
                options: [.skipsHiddenFiles])
            return try urls.filter {
                ["safetensors", "gguf"].contains($0.pathExtension.lowercased())
            }.reduce(into: 0) { total, url in
                let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
                let (next, overflow) = total.addingReportingOverflow(size)
                guard !overflow else { throw ConcurrentTextRuntimeError.memoryBudgetExceeded }
                total = next
            }
        }
        let target = try checkpointBytes(directory)
        let draft = try mtpCompanionDirectory.map(checkpointBytes) ?? 0
        let (checkpoint, checkpointOverflow) = target.addingReportingOverflow(draft)
        guard !checkpointOverflow, checkpoint > 0 else {
            throw ConcurrentTextRuntimeError.invalidConfiguration
        }
        let (runtimeReservation, runtimeOverflow) = configuration.prefixCacheBytes
            .addingReportingOverflow(configuration.workingMemoryBytes)
        guard !runtimeOverflow else { throw ConcurrentTextRuntimeError.memoryBudgetExceeded }
        // Scale scratch space with the artifact. A fixed desktop-sized floor would reject
        // valid tiny checkpoints whose complete configured budget is intentionally smaller.
        let loaderWorkspace = max(1 * 1024 * 1024, checkpoint / 10)
        let (base, baseOverflow) = checkpoint.addingReportingOverflow(runtimeReservation)
        let (required, requiredOverflow) = base.addingReportingOverflow(loaderWorkspace)
        guard !baseOverflow, !requiredOverflow else {
            throw ConcurrentTextRuntimeError.memoryBudgetExceeded
        }
        let physical = Int(clamping: ProcessInfo.processInfo.physicalMemory)
        let physicalCeiling = physical - physical / 5
        let gpuCeiling = GPU.maxRecommendedWorkingSetBytes() ?? physicalCeiling
        let deviceBudget = min(configuration.memoryBudgetBytes, min(physicalCeiling, gpuCeiling))
        return NativeTextModelMemoryAdmission(
            checkpointBytes: checkpoint, requiredBytes: required,
            deviceBudgetBytes: max(0, deviceBudget))
    }

    /// Constructs a raw runtime request using this model's exact chat tokens and stop IDs.
    /// String-stop checkpoints need a text-stream stop matcher and are explicitly rejected here.
    public func prepareRequest(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]? = nil,
        additionalContext: [String: any Sendable]? = nil,
        maximumOutputTokens: Int = 512, temperature: Float? = nil,
        topP: Float? = nil, topK: Int? = nil, seed: UInt64? = nil,
        priority: ConcurrentTextRuntime.Priority = .interactive, prefixTokenCount: Int = 0,
        cachePromptPrefix: Bool = false, promptLookup: Bool = false
    ) throws -> ConcurrentTextRuntime.Request {
        guard text.stopStrings.isEmpty else {
            throw CheckpointTextError.stringStopsRequireTextGeneration
        }
        guard maximumOutputTokens <= configuration.maxOutputTokens else {
            throw ConcurrentTextRuntimeError.invalidRequest
        }
        let tokens = try text.prepareChat(
            messages: messages, tools: tools, additionalContext: additionalContext)
        return try prepareRequest(
            tokens: tokens, maximumOutputTokens: maximumOutputTokens, temperature: temperature,
            topP: topP, topK: topK, seed: seed, priority: priority,
            prefixTokenCount: prefixTokenCount, cachePromptPrefix: cachePromptPrefix,
            promptLookup: promptLookup)
    }

    /// Constructs a runtime request from chat tokens rendered by this model's text processor.
    public func prepareRequest(
        tokens: [Int], maximumOutputTokens: Int = 512, temperature: Float? = nil,
        topP: Float? = nil, topK: Int? = nil, seed: UInt64? = nil,
        priority: ConcurrentTextRuntime.Priority = .interactive, prefixTokenCount: Int = 0,
        cachePromptPrefix: Bool = false, promptLookup: Bool = false
    ) throws -> ConcurrentTextRuntime.Request {
        guard text.stopStrings.isEmpty else {
            throw CheckpointTextError.stringStopsRequireTextGeneration
        }
        guard maximumOutputTokens <= configuration.maxOutputTokens else {
            throw ConcurrentTextRuntimeError.invalidRequest
        }
        try text.validate(tokens)
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
            prefixTokenCount: resolvedPrefixTokenCount, cacheIdentity: cacheIdentity,
            promptLookup: promptLookup)
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
            speculativeBlockSize: speculativeBlockSize,
            promptLookupDraftTokens: promptLookupDraftTokens)
    }
}
