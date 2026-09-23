// Copyright © 2026 Faux Fox.

import Foundation
import MLX
import MLXNN

func scheduledMTPVerificationBlockSize(
    requestedBlockSize: Int, drafter: any MTPDrafterModel, cache: [KVCache]
) -> Int {
    let drafterLimit = Swift.min(
        requestedBlockSize, drafter.maximumBlockSize ?? requestedBlockSize)
    return Swift.min(drafterLimit, MTPSpeculativeTokenIterator.maximumBlockSize(for: cache))
}

/// Opt-in contract for bounded, causal text forwards with all mutable state in the cache.
/// Conformers must support arbitrary chunk boundaries, return no out-of-cache state,
/// and keep weights read-only. Supported leaves are simple attention and Mamba caches.
public protocol ScheduledTextModel: LanguageModel {
    var vocabularySize: Int { get }
    /// Conservative bytes per token across all cache layers, including both K and V.
    var scheduledCacheBytesPerToken: Int { get }
    /// Recurrent-state bytes for one live boundary plus the requested verification boundaries.
    /// `verificationBlockSize` is one for ordinary decoding and includes the bonus token for MTP.
    func scheduledRecurrentStateBytes(forVerificationBlockSize verificationBlockSize: Int) -> Int
    /// Conservative recurrent-state reservation retained for source compatibility.
    var scheduledRecurrentStateBytes: Int { get }
    var scheduledSupportsBatchDecode: Bool { get }
    /// Bounded weight residency held outside registered model parameters.
    var scheduledAdditionalResidentWeightBytes: Int { get }
    /// A streamed model may need a host-side routing and I/O phase before MLX work.
    /// The runtime then keeps one request active so the model owns its expert slots.
    var scheduledRequiresExclusiveExecution: Bool { get }
    /// Upper bound for one scheduled forward. `nil` leaves prefill chunking unchanged.
    var scheduledMaximumForwardTokens: Int? { get }
    /// Whether this model's complete request state can be snapshotted with its KV cache.
    var scheduledSupportsPrefixCache: Bool { get }
    var scheduledMTPArchitectureID: String? { get }
    /// Request lifecycle hooks for models with transient state outside their KV cache.
    func scheduledBeginRequest() throws
    /// A prompt-aware lifecycle hook. Existing conformers may keep implementing
    /// `scheduledBeginRequest()`; the default bridge preserves that behavior.
    func scheduledBeginRequest(promptTokenCount: Int) throws
    func scheduledEndRequest()
    /// Whether the first generated token uses greedy selection without advancing the sampler.
    var scheduledFirstTokenGreedy: Bool { get }
    nonisolated(nonsending) func scheduledForward(
        _ tokens: MLXArray,
        cache: [KVCache]
    ) async throws -> MLXArray
    /// Returns only the final prefill token's logits when earlier rows are not needed.
    nonisolated(nonsending) func scheduledFinalPrefillForward(
        _ tokens: MLXArray,
        cache: [KVCache]
    ) async throws -> MLXArray
}

extension ScheduledTextModel {
    public func scheduledRecurrentStateBytes(forVerificationBlockSize _: Int) -> Int {
        scheduledRecurrentStateBytes
    }
    public var scheduledRecurrentStateBytes: Int { 0 }
    public var scheduledSupportsBatchDecode: Bool { false }
    public var scheduledAdditionalResidentWeightBytes: Int { 0 }
    public var scheduledRequiresExclusiveExecution: Bool { false }
    public var scheduledMaximumForwardTokens: Int? { nil }
    public var scheduledSupportsPrefixCache: Bool { true }
    public var scheduledMTPArchitectureID: String? { nil }
    public func scheduledBeginRequest() throws {}
    public func scheduledBeginRequest(promptTokenCount: Int) throws {
        try scheduledBeginRequest()
    }
    public func scheduledEndRequest() {}
    public var scheduledFirstTokenGreedy: Bool { false }
    public nonisolated(nonsending) func scheduledFinalPrefillForward(
        _ tokens: MLXArray,
        cache: [KVCache]
    ) async throws -> MLXArray {
        try await scheduledForward(tokens, cache: cache)
    }
}

/// Caller-supplied content identities, not mutable repository names or branch names.
public struct PrefixCacheIdentity: Hashable, Codable, Sendable {
    public let modelRevision: String
    public let tokenizerRevision: String
    public let chatTemplateRevision: String
    public let adapterRevision: String
    public let cacheLayoutRevision: String
    public let draftRevision: String
    public let quantizationContract: String
    public let transformContract: String

    public init(
        modelRevision: String, tokenizerRevision: String, chatTemplateRevision: String,
        adapterRevision: String, cacheLayoutRevision: String, draftRevision: String = "none",
        quantizationContract: String = "native", transformContract: String = "none"
    ) {
        self.modelRevision = modelRevision
        self.tokenizerRevision = tokenizerRevision
        self.chatTemplateRevision = chatTemplateRevision
        self.adapterRevision = adapterRevision
        self.cacheLayoutRevision = cacheLayoutRevision
        self.draftRevision = draftRevision
        self.quantizationContract = quantizationContract
        self.transformContract = transformContract
    }

    private enum CodingKeys: String, CodingKey {
        case modelRevision, tokenizerRevision, chatTemplateRevision, adapterRevision
        case cacheLayoutRevision, draftRevision, quantizationContract, transformContract
    }

    public init(from decoder: any Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        modelRevision = try values.decode(String.self, forKey: .modelRevision)
        tokenizerRevision = try values.decode(String.self, forKey: .tokenizerRevision)
        chatTemplateRevision = try values.decode(String.self, forKey: .chatTemplateRevision)
        adapterRevision = try values.decode(String.self, forKey: .adapterRevision)
        cacheLayoutRevision = try values.decode(String.self, forKey: .cacheLayoutRevision)
        draftRevision = try values.decodeIfPresent(String.self, forKey: .draftRevision) ?? "none"
        quantizationContract =
            try values.decodeIfPresent(String.self, forKey: .quantizationContract) ?? "native"
        transformContract =
            try values.decodeIfPresent(String.self, forKey: .transformContract) ?? "none"
    }
}

public enum ConcurrentTextRuntimeError: Error, Equatable {
    case invalidConfiguration
    case invalidRequest
    case unsupportedCache
    case queueFull
    case memoryBudgetExceeded
    case consumerTooSlow
    case shutDown
    case modelNotLoaded
}

/// A token-level inference service independent of chat sessions and transport.
///
/// One actor owns the model and every MLX array. Requests make interleaved progress
/// through bounded GPU forwards, with optional batched decode or speculative rounds.
public actor ConcurrentTextRuntime {
    public enum ExecutionMode: String, Codable, Sendable {
        case interleaved, batchedDecode, speculative, promptLookup
    }

    public struct CacheQuantization: Hashable, Codable, Sendable {
        public let bits: Int
        public let groupSize: Int
        public init(bits: Int = 8, groupSize: Int = 64) {
            self.bits = bits
            self.groupSize = groupSize
        }
    }

    public struct Configuration: Sendable {
        public let memoryBudgetBytes: Int
        public let prefixCacheBytes: Int
        /// Measured upper bound for transient forward/copy storage per active request.
        public let workingMemoryBytes: Int
        public let maxActiveRequests: Int
        public let maxQueuedRequests: Int
        public let maxPromptTokens: Int
        public let maxOutputTokens: Int
        public let prefillChunkSize: Int
        public let streamBufferSize: Int
        public let batchDecode: Bool
        public let interactiveReservedSlots: Int
        public let cacheQuantization: CacheQuantization?
        public let persistentCache: RuntimePersistentCacheConfiguration?
        public let speculativeAdaptation: SpeculativeDecodingAdaptation?
        /// Total target verification positions: one committed bonus plus draft tokens.
        public let speculativeBlockSize: Int
        /// Maximum prompt-derived candidates verified in one target forward.
        public let promptLookupDraftTokens: Int

        public init(
            memoryBudgetBytes: Int, prefixCacheBytes: Int, workingMemoryBytes: Int,
            maxActiveRequests: Int = 4, maxQueuedRequests: Int = 32,
            maxPromptTokens: Int = 32_768, maxOutputTokens: Int = 4_096,
            prefillChunkSize: Int = 128, streamBufferSize: Int = 256,
            batchDecode: Bool = true, interactiveReservedSlots: Int = 0,
            cacheQuantization: CacheQuantization? = nil,
            persistentCache: RuntimePersistentCacheConfiguration? = nil,
            speculativeAdaptation: SpeculativeDecodingAdaptation? = .init(),
            speculativeBlockSize: Int = 4, promptLookupDraftTokens: Int = 1
        ) {
            self.memoryBudgetBytes = memoryBudgetBytes
            self.prefixCacheBytes = prefixCacheBytes
            self.workingMemoryBytes = workingMemoryBytes
            self.maxActiveRequests = maxActiveRequests
            self.maxQueuedRequests = maxQueuedRequests
            self.maxPromptTokens = maxPromptTokens
            self.maxOutputTokens = maxOutputTokens
            self.prefillChunkSize = prefillChunkSize
            self.streamBufferSize = streamBufferSize
            self.batchDecode = batchDecode
            self.interactiveReservedSlots = interactiveReservedSlots
            self.cacheQuantization = cacheQuantization
            self.persistentCache = persistentCache
            self.speculativeAdaptation = speculativeAdaptation
            self.speculativeBlockSize = speculativeBlockSize
            self.promptLookupDraftTokens = promptLookupDraftTokens
        }
    }

    public enum Priority: Sendable {
        case interactive, background
    }

    public struct Request: Sendable {
        public let tokens: [Int]
        public let maxTokens: Int
        public let speculative: Bool
        public let promptLookup: Bool
        public let temperature: Float
        public let topP: Float
        public let topK: Int
        public let seed: UInt64?
        public let stopTokenIDs: Set<Int>
        public let priority: Priority
        /// Cache this many initial tokens. Must leave at least one prompt token to evaluate.
        /// MTP reuses the safe frontier whose one-token lookahead also lies within this prefix.
        public let prefixTokenCount: Int
        /// A mismatch disables reuse and publication for this request; generation stays cold.
        public let cacheIdentity: PrefixCacheIdentity?

        public init(
            tokens: [Int], maxTokens: Int, temperature: Float = 0,
            topP: Float = 1, topK: Int = 0, seed: UInt64? = nil,
            stopTokenIDs: Set<Int> = [], priority: Priority = .interactive,
            prefixTokenCount: Int = 0, cacheIdentity: PrefixCacheIdentity? = nil,
            speculative: Bool = true, promptLookup: Bool = false
        ) {
            self.speculative = speculative
            self.promptLookup = promptLookup
            self.tokens = tokens
            self.maxTokens = maxTokens
            self.temperature = temperature
            self.topP = topP
            self.topK = topK
            self.seed = seed
            self.stopTokenIDs = stopTokenIDs
            self.priority = priority
            self.prefixTokenCount = prefixTokenCount
            self.cacheIdentity = cacheIdentity
        }
    }

    public enum Event: Sendable {
        case admitted(reusedPrefixTokens: Int)
        case prefill(processedTokens: Int, totalTokens: Int)
        case token(Int)
        case finished(GenerateStopReason)
        case execution(ExecutionMode)
        case fallback(reason: String)
        case speculation(SpeculativeDecodingTelemetry)
    }

    public struct Generation: Sendable {
        public let id: UUID
        public let events: AsyncThrowingStream<Event, Error>
    }

    public struct Status: Sendable {
        public let activeRequests: Int
        public let queuedRequests: Int
        public let reservedBytes: Int
        public let cachedPrefixBytes: Int
        public let cachedPrefixes: Int
        public let batchedForwardCount: Int
        public let maximumBatchSize: Int
        public let persistentCacheFailures: Int
    }

    public struct Capabilities: Sendable {
        public let immutablePrefixSnapshots: Bool
        public let independentPrefixForks: Bool
        public let prefixRestore: Bool
        public let recurrentState: Bool
        public let speculativeDecoding: Bool
        public let speculativeLimitation: String?
        public let promptLookupDecoding: Bool
        public let fusedBatching: Bool
        public let executionMode: ExecutionMode
        public let limitation: String?
        public let parallelGPUExecution = false
    }

    public nonisolated let capabilities: Capabilities
    public nonisolated let identity: PrefixCacheIdentity
    private var ownedDrafter: (any IncrementalMTPDrafterModel)?
    private var ownedModel: (any ScheduledTextModel)?
    private var model: any ScheduledTextModel { ownedModel! }
    private let configuration: Configuration
    private let speculativeKVCacheConfiguration: KVCacheConfiguration?
    private let weightBytes: Int
    private let scheduledSpeculativeBlockSize: Int
    private let scheduledPromptLookupDraftTokens: Int
    private var active: [Slot] = []
    private var pending: [Slot] = []
    private var prefixes: [Prefix] = []
    private var running = false
    private var closed = false
    private var resources: InferenceResourceBudget?
    private var residentReservation: UUID?
    private var releasedReservations: [UUID] = []
    private var persistentStore: RuntimePersistentPrefixStore?
    private var persistentSpeculativeStore: RuntimePersistentPrefixStore?
    private var persistentCacheFailures = 0
    private var batchedForwardCount = 0
    private var maximumBatchSize = 1

    private typealias Continuation = AsyncThrowingStream<Event, Error>.Continuation

    private final class Slot {
        let id: UUID
        let request: Request
        let continuation: Continuation
        let reservation: Int
        let sampler: any LogitSampler
        var cache: [KVCache] = []
        var position = 0
        var generated = 0
        var lastToken: Int?
        var service: Double = 0
        var iterator: MTPSpeculativeTokenIterator?
        var lookup: PromptLookupCandidates?
        var lookupStorage: KVCacheStorage?
        var lookupRecentTokens: [Int] = []
        var lookupPendingTokens: [Int] = []
        var lookupPendingIndex = 0
        var lookupCommittedPendingTokenCount = 0
        var lookupNativeHybridRound = false
        var lookupTelemetry = SpeculativeDecodingTelemetry()
        var reportedMTPFallback: String?
        var modelRequestBegan = false
        var cancellationRequested = false
        var finishing = false
        var finished = false
        var inFlightForwardCount = 0
        var finishReason: GenerateStopReason?
        var finishError: Error?
        var settlementWaiters: [CheckedContinuation<Void, Never>] = []

        init(id: UUID, request: Request, continuation: Continuation, reservation: Int) {
            self.id = id
            self.request = request
            self.continuation = continuation
            self.reservation = reservation
            self.sampler = GenerateParameters(
                temperature: request.temperature,
                topP: request.topP,
                topK: request.topK,
                seed: request.seed
            ).sampler()
        }
    }

    private struct Prefix {
        enum State {
            case ordinary([KVCache])
            case speculative(MTPSpeculativeTokenIterator.ScheduledPrefix)
        }
        let tokens: [Int]
        let state: State
        let bytes: Int

        var speculative: Bool {
            if case .speculative = state { return true }
            return false
        }
    }

    /// Transfers exclusive model ownership. Load/quantize/apply adapters before this call.
    /// Release this runtime to unload weights; use `shutdown()` to settle requests first.
    public init(
        model: sending any ScheduledTextModel, identity: PrefixCacheIdentity,
        configuration: Configuration, drafter: sending (any IncrementalMTPDrafterModel)? = nil
    ) throws {
        guard configuration.memoryBudgetBytes > 0, configuration.prefixCacheBytes >= 0,
            configuration.workingMemoryBytes > 0, configuration.maxActiveRequests > 0,
            configuration.maxQueuedRequests > 0, configuration.maxPromptTokens > 0,
            configuration.maxOutputTokens > 0, configuration.prefillChunkSize > 0,
            configuration.streamBufferSize > 0, configuration.speculativeBlockSize >= 2,
            (1 ... 32).contains(configuration.promptLookupDraftTokens),
            model.scheduledCacheBytesPerToken > 0,
            model.scheduledAdditionalResidentWeightBytes >= 0,
            model.scheduledMaximumForwardTokens.map({ $0 > 0 }) ?? true,
            configuration.interactiveReservedSlots >= 0,
            configuration.interactiveReservedSlots < configuration.maxActiveRequests,
            model.vocabularySize > 0
        else { throw ConcurrentTextRuntimeError.invalidConfiguration }
        guard !(model.scheduledRequiresExclusiveExecution && drafter != nil) else {
            throw ConcurrentTextRuntimeError.invalidConfiguration
        }
        model.train(false)
        try model.prepare()
        eval(model)
        if let drafter {
            guard drafter.targetArchitectureID == model.scheduledMTPArchitectureID else {
                throw ConcurrentTextRuntimeError.invalidConfiguration
            }
            drafter.train(false)
            eval(drafter)
        }
        let registeredWeights =
            model.parameters().flattened().reduce(0) { $0 + $1.1.nbytes }
            + (drafter?.parameters().flattened().reduce(0) { $0 + $1.1.nbytes } ?? 0)
        let (weights, overflow) = registeredWeights.addingReportingOverflow(
            model.scheduledAdditionalResidentWeightBytes)
        guard !overflow, weights < configuration.memoryBudgetBytes,
            configuration.prefixCacheBytes < configuration.memoryBudgetBytes - weights,
            configuration.workingMemoryBytes
                < configuration.memoryBudgetBytes - weights - configuration.prefixCacheBytes
        else { throw ConcurrentTextRuntimeError.memoryBudgetExceeded }
        let cache = try model.newCache(parameters: nil)
        guard !cache.isEmpty,
            cache.allSatisfy({
                type(of: $0) == KVCacheSimple.self || type(of: $0) == MambaCache.self
            })
        else {
            throw ConcurrentTextRuntimeError.unsupportedCache
        }
        if let quantization = configuration.cacheQuantization {
            guard [4, 8].contains(quantization.bits),
                [32, 64, 128].contains(quantization.groupSize)
            else {
                throw ConcurrentTextRuntimeError.invalidConfiguration
            }
        }
        let speculativeKVCacheConfiguration = try configuration.cacheQuantization.map {
            KVCacheConfiguration(
                strategy: .affine(
                    try AffineKVCacheConfiguration(
                        bits: $0.bits, groupSize: $0.groupSize, compressionStart: 0)),
                compatibility: .allowPartial)
        }
        if let adaptation = configuration.speculativeAdaptation {
            guard adaptation.minimumDraftTokens > 0,
                adaptation.minimumAcceptanceRate.isFinite,
                (0 ... 1).contains(adaptation.minimumAcceptanceRate)
            else { throw ConcurrentTextRuntimeError.invalidConfiguration }
        }
        let scheduledSpeculativeBlockSize: Int
        if let drafter {
            scheduledSpeculativeBlockSize = scheduledMTPVerificationBlockSize(
                requestedBlockSize: configuration.speculativeBlockSize,
                drafter: drafter, cache: cache)
        } else {
            scheduledSpeculativeBlockSize = 1
        }
        self.ownedModel = model
        self.ownedDrafter = drafter
        self.identity = identity
        self.configuration = configuration
        self.speculativeKVCacheConfiguration = speculativeKVCacheConfiguration
        self.weightBytes = weights
        self.scheduledSpeculativeBlockSize = scheduledSpeculativeBlockSize
        let hasRecurrentCache = cache.contains { $0 is MambaCache }
        let hybridRewind =
            (model as? any PromptLookupHybridModel)?.maximumNativeTargetCacheRewind ?? 0
        self.scheduledPromptLookupDraftTokens =
            hasRecurrentCache
            ? min(configuration.promptLookupDraftTokens, hybridRewind)
            : configuration.promptLookupDraftTokens
        let streamed = model.scheduledRequiresExclusiveExecution
        let batching = configuration.batchDecode && model.scheduledSupportsBatchDecode && !streamed
        let simpleLookup = cache.allSatisfy { type(of: $0) == KVCacheSimple.self }
        let hybridLookup =
            hasRecurrentCache && hybridRewind >= 1
            && cache.allSatisfy {
                type(of: $0) == KVCacheSimple.self || type(of: $0) == MambaCache.self
            }
        let promptLookup =
            !streamed && configuration.cacheQuantization == nil
            && (model.scheduledMaximumForwardTokens.map { $0 >= 2 } ?? true)
            && (simpleLookup || hybridLookup)
        self.capabilities = Capabilities(
            immutablePrefixSnapshots: model.scheduledSupportsPrefixCache,
            independentPrefixForks: model.scheduledSupportsPrefixCache,
            prefixRestore: model.scheduledSupportsPrefixCache,
            recurrentState: cache.contains { $0 is MambaCache },
            speculativeDecoding: drafter != nil && !streamed,
            speculativeLimitation: streamed
                ? "Streamed experts only support direct scheduled execution"
                : drafter == nil
                    ? "No matching trained MTP head supplied"
                    : drafter is any ScheduledMTPPrefixCachingDrafter
                        ? drafter!.requiresSharedTargetKV
                            ? "MTP in-memory and disk prefix restore are available; shared-KV verification is interleaved"
                            : "MTP in-memory and disk prefix restore and batched verification are available"
                        : "MTP prefix restore and batched verification are unavailable",
            promptLookupDecoding: promptLookup,
            fusedBatching: batching, executionMode: batching ? .batchedDecode : .interleaved,
            limitation: streamed
                ? "Streamed experts serialize ownership and disable batch decode"
                : batching
                    ? nil : "Model has not opted into batched projection/row-native attention")
        if let persistent = configuration.persistentCache {
            let layout =
                cache.map { String(describing: type(of: $0)) }.joined(separator: ",")
                + ":" + String(describing: configuration.cacheQuantization)
            self.persistentStore = try RuntimePersistentPrefixStore(
                configuration: persistent, identity: identity, layoutFingerprint: layout)
            if let drafter = drafter as? any ScheduledMTPPrefixCachingDrafter {
                let draftCache = drafter.makeState(parameters: nil).cache
                let speculativeLayout =
                    layout + "/speculative:"
                    + draftCache.map { String(describing: type(of: $0)) }.joined(separator: ",")
                self.persistentSpeculativeStore = try RuntimePersistentPrefixStore(
                    configuration: persistent, identity: identity,
                    layoutFingerprint: speculativeLayout)
            }
        }
    }

    public func generate(_ request: Request) async throws -> Generation {
        guard !closed else { throw ConcurrentTextRuntimeError.shutDown }
        try Task.checkCancellation()
        guard !request.tokens.isEmpty, request.tokens.count <= configuration.maxPromptTokens,
            request.maxTokens > 0, request.maxTokens <= configuration.maxOutputTokens,
            request.temperature.isFinite, request.temperature >= 0,
            request.topP.isFinite, request.topP > 0, request.topP <= 1,
            request.topK >= 0,
            request.prefixTokenCount >= 0, request.prefixTokenCount < request.tokens.count,
            request.tokens.allSatisfy({ $0 >= 0 && $0 < model.vocabularySize })
        else { throw ConcurrentTextRuntimeError.invalidRequest }
        guard pending.count < configuration.maxQueuedRequests else {
            throw ConcurrentTextRuntimeError.queueFull
        }
        let (length, overflow1) = request.tokens.count.addingReportingOverflow(request.maxTokens)
        let (padded, overflow2) = length.addingReportingOverflow(255)
        let (kvBytes, overflow3) = padded.multipliedReportingOverflow(
            by: model.scheduledCacheBytesPerToken
                + (usesMTP(request) ? ownedDrafter!.cacheBytesPerToken : 0))
        let verificationBlockSize =
            usesMTP(request)
            ? scheduledSpeculativeBlockSize
            : usesPromptLookup(request) ? scheduledPromptLookupDraftTokens + 1 : 1
        let (stateBytes, stateOverflow) = kvBytes.addingReportingOverflow(
            model.scheduledRecurrentStateBytes(
                forVerificationBlockSize: verificationBlockSize))
        let (reservation, overflow4) = stateBytes.addingReportingOverflow(
            configuration.workingMemoryBytes)
        guard !overflow1, !overflow2, !overflow3, !stateOverflow, !overflow4,
            reservation <= requestBudget
        else {
            throw ConcurrentTextRuntimeError.memoryBudgetExceeded
        }
        if let resources,
            !(await resources.canEverFit(reservation, background: request.priority == .background))
        {
            throw ConcurrentTextRuntimeError.memoryBudgetExceeded
        }
        guard !closed else { throw ConcurrentTextRuntimeError.shutDown }
        try Task.checkCancellation()
        guard pending.count < configuration.maxQueuedRequests else {
            throw ConcurrentTextRuntimeError.queueFull
        }
        let id = UUID()
        let (stream, continuation) = AsyncThrowingStream<Event, Error>.makeStream(
            bufferingPolicy: .bufferingOldest(configuration.streamBufferSize))
        continuation.onTermination = { [weak self] termination in
            if case .cancelled = termination {
                Task { await self?.cancel(id) }
            }
        }
        pending.append(
            Slot(id: id, request: request, continuation: continuation, reservation: reservation))
        if !running {
            running = true
            Task { await self.run() }
        }
        await resources?.signal()
        return Generation(id: id, events: stream)
    }

    /// Returns only after this request's last bounded forward has settled and state is released.
    /// Call explicitly when breaking stream iteration early.
    public func cancel(_ id: UUID) async {
        if let slot = pending.first(where: { $0.id == id }) ?? active.first(where: { $0.id == id })
        {
            finish(slot, reason: .cancelled)
            await waitForSettlement(of: slot)
        }
        await flushReleases()
        await resources?.signal()
    }

    public func clearPrefixCache() {
        prefixes.removeAll()
    }

    public func clearCaches(persistent: Bool = false) throws {
        clearPrefixCache()
        if persistent {
            try persistentStore?.clear()
            try persistentSpeculativeStore?.clear()
        }
    }

    public func shutdown() async {
        closed = true
        let slots = pending + active
        for slot in slots { finish(slot, reason: .cancelled) }
        for slot in slots { await waitForSettlement(of: slot) }
        clearPrefixCache()
        await flushReleases()
        if let residentReservation {
            self.residentReservation = nil
            await resources?.unregister(residentReservation)
        }
        ownedModel = nil
        ownedDrafter = nil
    }

    public func status() -> Status {
        Status(
            activeRequests: active.count, queuedRequests: pending.count,
            reservedBytes: (closed ? 0 : weightBytes + configuration.prefixCacheBytes)
                + activeBytes,
            cachedPrefixBytes: prefixBytes, cachedPrefixes: prefixes.count,
            batchedForwardCount: batchedForwardCount, maximumBatchSize: maximumBatchSize,
            persistentCacheFailures: persistentCacheFailures)
    }

    func attach(resources: InferenceResourceBudget, reservation: UUID, maximumResidentBytes: Int)
        async throws
    {
        guard !closed, !running, self.resources == nil,
            weightBytes <= maximumResidentBytes - configuration.prefixCacheBytes
        else {
            throw ConcurrentTextRuntimeError.memoryBudgetExceeded
        }
        self.resources = resources
        self.residentReservation = reservation
        try await resources.register(
            reservation, bytes: weightBytes + configuration.prefixCacheBytes)
    }

    private func flushReleases() async {
        let released = releasedReservations
        releasedReservations.removeAll()
        for id in released { await resources?.release(id) }
    }

    private func waitForSettlement(of slot: Slot) async {
        guard !slot.finished else { return }
        await withCheckedContinuation { slot.settlementWaiters.append($0) }
    }

    private func beginForward(for slots: [Slot]) {
        for slot in slots { slot.inFlightForwardCount += 1 }
    }

    private func completedForward(for slot: Slot) -> Bool {
        precondition(slot.inFlightForwardCount > 0)
        slot.inFlightForwardCount -= 1
        return slot.finishing && slot.inFlightForwardCount == 0
    }

    private var requestBudget: Int {
        configuration.memoryBudgetBytes - weightBytes - configuration.prefixCacheBytes
    }
    private var activeBytes: Int { active.reduce(0) { $0 + $1.reservation } }
    private var prefixBytes: Int { prefixes.reduce(0) { $0 + $1.bytes } }

    private func run() async {
        while !pending.isEmpty || !active.isEmpty {
            let version = await resources?.version()
            guard !closed else { break }
            await admit()
            if let slot = active.min(by: { $0.service < $1.service }) {
                if let resources {
                    guard
                        await resources.beginTurn(
                            slot.id, background: slot.request.priority == .background)
                    else {
                        await flushReleases()
                        continue
                    }
                    guard !closed, active.contains(where: { $0.id == slot.id }) else {
                        await resources.endTurn(slot.id)
                        continue
                    }
                }
                var selected = [slot]
                do {
                    if capabilities.fusedBatching, slot.iterator == nil,
                        !promptLookupReady(slot),
                        slot.position == slot.request.tokens.count
                    {
                        selected += active.filter {
                            $0.id != slot.id && $0.iterator == nil && !promptLookupReady($0)
                                && $0.position == $0.request.tokens.count
                                && $0.service <= slot.service + 1
                        }
                    } else if capabilities.fusedBatching,
                        let width = slot.iterator?.scheduledBatchVerificationWidth,
                        slot.position == slot.request.tokens.count
                    {
                        selected += active.filter {
                            $0.id != slot.id
                                && $0.position == $0.request.tokens.count
                                && $0.iterator?.scheduledBatchVerificationWidth == width
                                && $0.service <= slot.service + 1
                        }
                    }
                    if selected.count > 1 {
                        if selected[0].iterator == nil {
                            try await batchStep(selected)
                        } else {
                            try batchSpeculativeStep(selected)
                        }
                    } else {
                        try await step(slot)
                    }
                    for item in selected where !item.finishing {
                        item.service += item.request.priority == .interactive ? 1 : 3
                    }
                } catch {
                    // Drain any graph submitted by a throwing backend before releasing its slot.
                    Stream().synchronize()
                    for item in selected { finish(item, error: error) }
                }
                await resources?.endTurn(slot.id)
            }
            await flushReleases()
            if active.isEmpty, !pending.isEmpty, let resources, let version {
                await resources.waitForChange(after: version)
            }
            await Task.yield()
        }
        running = false
    }

    private func admit() async {
        // FIFO admission reserves room for large requests instead of continually bypassing them.
        let maximumActive =
            model.scheduledRequiresExclusiveExecution ? 1 : configuration.maxActiveRequests
        while let slot = pending.first, active.count < maximumActive {
            let backgroundSlotsFull =
                active.filter { $0.request.priority == .background }.count
                >= configuration.maxActiveRequests - configuration.interactiveReservedSlots
            if slot.request.priority == .background,
                backgroundSlotsFull || slot.reservation > requestBudget - activeBytes
            {
                if let index = pending.firstIndex(where: { $0.request.priority == .interactive }) {
                    let interactive = pending.remove(at: index)
                    pending.insert(interactive, at: 0)
                    continue
                }
                break
            }
            guard slot.reservation <= requestBudget - activeBytes else { break }
            if let resources {
                let acquired: Bool
                do {
                    acquired = try await resources.acquire(
                        slot.id, bytes: slot.reservation,
                        background: slot.request.priority == .background)
                } catch {
                    finish(slot, error: error)
                    continue
                }
                guard !closed, pending.first?.id == slot.id else {
                    await resources.release(slot.id)
                    continue
                }
                if !acquired {
                    if slot.request.priority == .background,
                        let index = pending.firstIndex(where: {
                            $0.request.priority == .interactive
                        })
                    {
                        let interactive = pending.remove(at: index)
                        pending.insert(interactive, at: 0)
                        continue
                    }
                    break
                }
            }
            pending.removeFirst()
            do {
                slot.cache = try model.newCache(parameters: nil)
                guard
                    slot.cache.allSatisfy({
                        type(of: $0) == KVCacheSimple.self || type(of: $0) == MambaCache.self
                    })
                else {
                    throw ConcurrentTextRuntimeError.unsupportedCache
                }
                if usesMTP(slot.request), let drafter = ownedDrafter {
                    let prefix = takePrefix(for: slot.request, speculative: true)
                    var snapshot: MTPSpeculativeTokenIterator.ScheduledPrefix?
                    if case .speculative(let state) = prefix?.state {
                        snapshot = state
                    } else {
                        snapshot = nil
                    }
                    if snapshot == nil, model.scheduledSupportsPrefixCache,
                        slot.request.cacheIdentity == identity,
                        let persistentSpeculativeStore,
                        let cachingDrafter = drafter as? any ScheduledMTPPrefixCachingDrafter
                    {
                        do {
                            let draftPrototype = cachingDrafter.makeState(parameters: nil).cache
                            if let restored = try persistentSpeculativeStore.restore(
                                prompt: slot.request.tokens,
                                maximumPrefixTokens: slot.request.prefixTokenCount,
                                prototype: slot.cache + draftPrototype)
                            {
                                let mainCount = slot.cache.count
                                snapshot = .init(
                                    mainCache: Array(restored.cache[..<mainCount]),
                                    drafterCache: Array(restored.cache[mainCount...]),
                                    processedTokenCount: restored.processedTokenCount)
                            }
                        } catch { persistentCacheFailures += 1 }
                    }
                    slot.iterator = try MTPSpeculativeTokenIterator(
                        scheduledPrompt: slot.request.tokens, mainModel: model, drafter: drafter,
                        mainCache: slot.cache,
                        parameters: GenerateParameters(
                            maxTokens: slot.request.maxTokens,
                            kvCache: speculativeKVCacheConfiguration,
                            temperature: slot.request.temperature,
                            topP: slot.request.topP,
                            topK: slot.request.topK,
                            seed: slot.request.seed),
                        blockSize: configuration.speculativeBlockSize, prefix: snapshot,
                        adaptation: configuration.speculativeAdaptation
                    )
                    slot.cache = slot.iterator!.mainCache
                    slot.position = snapshot?.processedTokenCount ?? 0
                } else if let prefix = takePrefix(for: slot.request, speculative: false),
                    case .ordinary(let cache) = prefix.state
                {
                    slot.cache = cache.map { $0.copy() }
                    eval(slot.cache.flatMap { $0.innerState() })
                    slot.position = prefix.tokens.count
                } else if model.scheduledSupportsPrefixCache,
                    slot.request.cacheIdentity == identity, let persistentStore
                {
                    do {
                        if let restored = try persistentStore.restore(
                            prompt: slot.request.tokens,
                            maximumPrefixTokens: slot.request.prefixTokenCount,
                            prototype: slot.cache)
                        {
                            slot.cache = restored.cache
                            slot.position = restored.tokens.count
                        }
                    } catch { persistentCacheFailures += 1 }
                }
                if usesPromptLookup(slot.request) {
                    slot.lookup = PromptLookupCandidates(
                        prompt: slot.request.tokens, maximumNgramSize: 4)
                }
                try model.scheduledBeginRequest(promptTokenCount: slot.request.tokens.count)
                slot.modelRequestBegan = true
                slot.service = active.map(\.service).min() ?? 0
                active.append(slot)
                try emit(.admitted(reusedPrefixTokens: slot.position), to: slot)
                try emit(
                    .execution(
                        slot.iterator != nil
                            ? .speculative
                            : slot.lookup != nil ? .promptLookup : capabilities.executionMode),
                    to: slot)
                if slot.request.promptLookup, slot.lookup == nil {
                    try emit(
                        .fallback(
                            reason: slot.request.temperature == 0
                                ? "Prompt lookup is unavailable for this cache or model"
                                : "Prompt lookup requires greedy sampling"), to: slot)
                }
                if slot.request.speculative, ownedDrafter != nil, slot.iterator == nil,
                    slot.lookup == nil
                {
                    try emit(
                        .fallback(reason: "MTP is unavailable for this request"), to: slot)
                }
                if slot.iterator != nil, slot.request.prefixTokenCount > 0 {
                    if !(ownedDrafter is any ScheduledMTPPrefixCachingDrafter) {
                        try emit(
                            .fallback(reason: "This MTP drafter requires cold prompt caches"),
                            to: slot)
                    }
                }
            } catch {
                finish(slot, error: error)
            }
        }
    }

    private func usesMTP(_ request: Request) -> Bool {
        request.speculative && capabilities.speculativeDecoding && !usesPromptLookup(request)
    }

    private func usesPromptLookup(_ request: Request) -> Bool {
        request.promptLookup && request.temperature == 0 && capabilities.promptLookupDecoding
    }

    private func promptLookupReady(_ slot: Slot) -> Bool {
        guard let lookup = slot.lookup else { return false }
        if slot.lookupPendingIndex < slot.lookupPendingTokens.count { return true }
        let remaining = slot.request.maxTokens - slot.generated
        guard remaining > 1 else { return false }
        return !lookup.continuation(
            after: slot.lookupRecentTokens,
            limit: min(scheduledPromptLookupDraftTokens, remaining - 1)
        ).isEmpty
    }

    private func takePrefix(for request: Request, speculative: Bool) -> Prefix? {
        guard model.scheduledSupportsPrefixCache, request.cacheIdentity == identity,
            let index = prefixes.indices.filter({
                prefixes[$0].speculative == speculative
                    && prefixes[$0].tokens.count <= request.prefixTokenCount
                    && request.tokens.starts(with: prefixes[$0].tokens)
            }).max(by: { prefixes[$0].tokens.count < prefixes[$1].tokens.count })
        else { return nil }
        let prefix = prefixes.remove(at: index)
        prefixes.append(prefix)
        return prefix
    }

    private func promptLookupStep(_ slot: Slot) async throws -> Bool {
        if slot.lookupPendingIndex < slot.lookupPendingTokens.count {
            try emitPromptLookupPending(slot)
            return true
        }
        slot.lookupPendingTokens.removeAll(keepingCapacity: true)
        slot.lookupPendingIndex = 0
        slot.lookupCommittedPendingTokenCount = 0
        slot.lookupNativeHybridRound = false
        discardSpeculativePromptCacheCheckpoints(slot.cache)

        guard let lookup = slot.lookup, let lastToken = slot.lastToken else { return false }
        let remaining = slot.request.maxTokens - slot.generated
        guard remaining > 1 else { return false }
        let proposals = lookup.continuation(
            after: slot.lookupRecentTokens,
            limit: min(scheduledPromptLookupDraftTokens, remaining - 1))
        guard !proposals.isEmpty else { return false }

        let storage = slot.lookupStorage ?? KVCacheStorage(slot.cache, plan: .disabled)
        slot.lookupStorage = storage
        if slot.cache.contains(where: { $0 is MambaCache }) {
            try promptLookupHybridStep(
                slot, storage: storage, lastToken: lastToken, proposals: proposals)
            return true
        }
        guard let round = storage.beginRound(maximumPositions: proposals.count + 1) else {
            try disablePromptLookup(slot, reason: "Target cache cannot stage prompt lookup")
            return false
        }

        let input = MLXArray([lastToken] + proposals).expandedDimensions(axis: 0)
        beginForward(for: [slot])
        let logits: MLXArray
        do {
            logits = try await model.scheduledForward(input, cache: round.caches)
        } catch {
            let finishing = completedForward(for: slot)
            Stream().synchronize()
            storage.rollback(round)
            if finishing { settle(slot) }
            throw error
        }
        eval([logits] + round.caches.flatMap { $0.innerState() })
        if completedForward(for: slot) {
            storage.rollback(round)
            Stream().synchronize()
            settle(slot)
            return true
        }

        let width = proposals.count + 1
        guard round.writtenPositions == width, logits.ndim == 3,
            logits.dim(0) == 1, logits.dim(1) >= width
        else {
            storage.rollback(round)
            try disablePromptLookup(slot, reason: "Target did not return every verification row")
            return false
        }
        let targets = argMax(logits[0, 0 ..< width, 0...], axis: -1)
        eval(targets)
        let values = targets.asArray(Int.self)
        var accepted = 0
        while accepted < proposals.count, values[accepted] == proposals[accepted] {
            accepted += 1
        }
        storage.commit(round, retaining: accepted + 1)
        slot.lookupPendingTokens = Array(values.prefix(accepted + 1))
        slot.lookupCommittedPendingTokenCount = accepted
        slot.lookupTelemetry.recordRound(
            drafted: proposals.count, accepted: accepted, targetVerified: width,
            draftModelCalls: 0)
        let bytes = slot.cache.flatMap { $0.innerState() }.reduce(0) { $0 + $1.nbytes }
        guard bytes <= slot.reservation - configuration.workingMemoryBytes else {
            throw ConcurrentTextRuntimeError.memoryBudgetExceeded
        }
        try emitPromptLookupPending(slot)
        return true
    }

    private func promptLookupHybridStep(
        _ slot: Slot, storage: KVCacheStorage, lastToken: Int, proposals: [Int]
    ) throws {
        let width = proposals.count + 1
        var state = LMOutput.State()
        state[speculativeCheckpointOnlyKey] = true
        if proposals.count == 1 {
            state[mtpCacheCheckpointIndexKey] = 1
        } else {
            state[mtpCacheCheckpointIndicesKey] = Array(1 ... proposals.count)
        }
        let input = LMInput.Text(tokens: MLXArray([lastToken] + proposals))
        let output = model(input[text: .newAxis], cache: slot.cache, state: state)
        eval([output.logits] + slot.cache.flatMap { $0.innerState() })
        precondition(
            output.logits.ndim == 3 && output.logits.dim(0) == 1
                && output.logits.dim(1) >= width,
            "Hybrid prompt-lookup target must return every verification row")

        let targets = argMax(output.logits[0, 0 ..< width, 0...], axis: -1)
        eval(targets)
        let values = targets.asArray(Int.self)
        var accepted = 0
        while accepted < proposals.count, values[accepted] == proposals[accepted] {
            accepted += 1
        }
        let rejected = proposals.count - accepted
        if rejected > 0 {
            let rewound = rewindSpeculativePromptCache(slot.cache, numTokens: rejected)
            precondition(
                rewound == rejected,
                "Hybrid prompt-lookup target did not retain its recurrent checkpoint")
        }
        storage.commitProcessedTokens(accepted + 1)
        slot.lookupPendingTokens = Array(values.prefix(accepted + 1))
        slot.lookupCommittedPendingTokenCount = accepted
        slot.lookupNativeHybridRound = true
        slot.lookupTelemetry.recordRound(
            drafted: proposals.count, accepted: accepted, targetVerified: width,
            draftModelCalls: 0)
        let bytes = slot.cache.flatMap { $0.innerState() }.reduce(0) { $0 + $1.nbytes }
        guard bytes <= slot.reservation - configuration.workingMemoryBytes else {
            throw ConcurrentTextRuntimeError.memoryBudgetExceeded
        }
        try emitPromptLookupPending(slot)
    }

    private func emitPromptLookupPending(_ slot: Slot) throws {
        let token = slot.lookupPendingTokens[slot.lookupPendingIndex]
        slot.lookupPendingIndex += 1
        if slot.lookupNativeHybridRound {
            let consumed = min(
                slot.lookupPendingIndex, slot.lookupCommittedPendingTokenCount)
            let lookahead = slot.lookupCommittedPendingTokenCount - consumed
            for case let cache as MambaCache in slot.cache {
                if lookahead == 0 {
                    cache.discardSpeculativeCheckpoint()
                } else {
                    cache.retainSpeculativeCheckpoints(rewindingAtMost: lookahead)
                }
            }
        }
        slot.lookupTelemetry.recordGeneratedToken()
        try accept(token, slot: slot, fromPromptLookup: true)
    }

    private func disablePromptLookup(_ slot: Slot, reason: String) throws {
        slot.lookup = nil
        slot.lookupStorage = nil
        try emit(.execution(capabilities.executionMode), to: slot)
        try emit(.fallback(reason: reason), to: slot)
    }

    private func step(_ slot: Slot) async throws {
        if slot.iterator != nil {
            try speculativeStep(slot)
            return
        }
        if slot.lookup != nil, slot.position == slot.request.tokens.count,
            try await promptLookupStep(slot)
        {
            return
        }
        let request = slot.request
        let isPrefill = slot.position < request.tokens.count
        let tokens: [Int]
        if isPrefill {
            let remaining = request.tokens.count - slot.position
            var count = min(configuration.prefillChunkSize, remaining)
            if let maximum = model.scheduledMaximumForwardTokens {
                count = min(count, maximum)
            }
            if request.prefixTokenCount > slot.position {
                count = min(count, request.prefixTokenCount - slot.position)
            }
            tokens = Array(request.tokens[slot.position ..< slot.position + count])
        } else {
            guard let token = slot.lastToken else {
                throw ConcurrentTextRuntimeError.invalidRequest
            }
            tokens = [token]
        }
        let needsLogits = !isPrefill || slot.position + tokens.count == request.tokens.count
        let input = MLXArray(tokens).expandedDimensions(axis: 0)
        let logits: MLXArray
        beginForward(for: [slot])
        do {
            if isPrefill {
                logits = try await model.scheduledFinalPrefillForward(input, cache: slot.cache)
            } else {
                logits = try await model.scheduledForward(input, cache: slot.cache)
            }
        } catch {
            if completedForward(for: slot) {
                Stream().synchronize()
                settle(slot)
            }
            throw error
        }
        if completedForward(for: slot) {
            Stream().synchronize()
            settle(slot)
            return
        }
        // Intermediate chunks only contribute cache state; their vocabulary scores are unused.
        eval((needsLogits ? [logits] : []) + slot.cache.flatMap { $0.innerState() })
        if !isPrefill { slot.lookupStorage?.commitProcessedTokens(tokens.count) }
        try compress(slot)
        let bytes = slot.cache.flatMap { $0.innerState() }.reduce(0) { $0 + $1.nbytes }
        guard bytes <= slot.reservation - configuration.workingMemoryBytes else {
            throw ConcurrentTextRuntimeError.memoryBudgetExceeded
        }
        if isPrefill {
            slot.position += tokens.count
            try emit(
                .prefill(processedTokens: slot.position, totalTokens: request.tokens.count),
                to: slot)
            if slot.position == request.prefixTokenCount, request.cacheIdentity == identity {
                checkpoint(slot, bytes: bytes)
            }
            if slot.position < request.tokens.count { return }
        }
        try sample(logits, slot: slot)
    }

    private func speculativeStep(_ slot: Slot) throws {
        if slot.position < slot.request.tokens.count {
            let prefixFrontier = slot.request.prefixTokenCount - 1
            var end = min(slot.position + configuration.prefillChunkSize, slot.request.tokens.count)
            if slot.position < prefixFrontier, end > prefixFrontier {
                end = prefixFrontier
            }
            try slot.iterator!.prepareScheduledChunk(
                Array(slot.request.tokens[slot.position ..< end]),
                nextPromptToken: end < slot.request.tokens.count ? slot.request.tokens[end] : nil)
            guard
                slot.iterator!.scheduledResidentArrays.reduce(0, { $0 + $1.nbytes })
                    <= slot.reservation
            else { throw ConcurrentTextRuntimeError.memoryBudgetExceeded }
            slot.position = end
            try emit(
                .prefill(processedTokens: end, totalTokens: slot.request.tokens.count), to: slot)
            if slot.request.prefixTokenCount > 1,
                end == prefixFrontier,
                slot.request.cacheIdentity == identity
            {
                checkpointSpeculative(slot)
            }
            if end < slot.request.tokens.count { return }
        }
        guard let token = slot.iterator!.next() else {
            finish(slot, reason: .length)
            return
        }
        if let reason = slot.iterator!.passthroughReason, reason != slot.reportedMTPFallback {
            slot.reportedMTPFallback = reason
            try emit(.execution(.interleaved), to: slot)
            try emit(.fallback(reason: reason), to: slot)
        }
        let arrays = slot.iterator!.scheduledResidentArrays
        eval(arrays)
        guard arrays.reduce(0, { $0 + $1.nbytes }) <= slot.reservation else {
            throw ConcurrentTextRuntimeError.memoryBudgetExceeded
        }
        try accept(token, slot: slot)
    }

    private func sample(_ logits: MLXArray, slot: Slot) throws {
        let finalLogits = logits[0..., -1, 0...]
        let token =
            slot.generated == 0 && model.scheduledFirstTokenGreedy
            ? argMax(finalLogits, axis: -1).item(Int.self)
            : slot.sampler.sample(logits: finalLogits).item(Int.self)
        try accept(token, slot: slot)
    }

    private func accept(_ token: Int, slot: Slot, fromPromptLookup: Bool = false) throws {
        let request = slot.request
        guard token >= 0, token < model.vocabularySize else {
            throw ConcurrentTextRuntimeError.invalidRequest
        }
        if request.stopTokenIDs.contains(token) {
            slot.iterator?.discardGeneratedToken()
            if fromPromptLookup { slot.lookupTelemetry.discardGeneratedToken() }
            finish(slot, reason: .stop)
            return
        }
        slot.lastToken = token
        slot.generated += 1
        if slot.lookup != nil {
            slot.lookupRecentTokens.append(token)
            if slot.lookupRecentTokens.count > 4 { slot.lookupRecentTokens.removeFirst() }
        }
        try emit(.token(token), to: slot)
        if slot.generated == request.maxTokens { finish(slot, reason: .length) }
    }

    private func compress(_ slot: Slot) throws {
        if let quantization = configuration.cacheQuantization {
            for index in slot.cache.indices {
                if let cache = slot.cache[index] as? KVCacheSimple {
                    slot.cache[index] = try cache.toQuantized(
                        groupSize: quantization.groupSize, bits: quantization.bits)
                }
            }
            eval(slot.cache.flatMap { $0.innerState() })
        }
    }

    private func batchSpeculativeStep(_ slots: [Slot]) throws {
        guard let width = slots.first?.iterator?.scheduledBatchVerificationWidth,
            width >= 2
        else { throw ConcurrentTextRuntimeError.invalidRequest }

        var prepared = [MTPSpeculativeTokenIterator.ScheduledVerification]()
        prepared.reserveCapacity(slots.count)
        for slot in slots {
            guard var iterator = slot.iterator,
                iterator.scheduledBatchVerificationWidth == width,
                let verification = iterator.prepareScheduledVerification(maximumWidth: width)
            else { throw ConcurrentTextRuntimeError.invalidRequest }
            slot.iterator = iterator
            prepared.append(verification)
        }

        let batch = try RuntimeBatchCache(rows: prepared.map(\.cache))
        let inputs = concatenated(
            prepared.map { $0.tokens.expandedDimensions(axis: 0) }, axis: 0)
        var state = LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpHiddenLayerIDsKey] = prepared[0].hiddenLayerIDs
        if width == 2 {
            state[mtpCacheCheckpointIndexKey] = 1
        } else {
            state[mtpCacheCheckpointIndicesKey] = Array(1 ..< width)
        }

        beginForward(for: slots)
        let output = model(LMInput.Text(tokens: inputs), cache: batch.cache, state: state)
        eval(
            [output.logits]
                + (output.state?[mtpLastHiddenStatesKey].map { [$0] } ?? [])
                + batch.cache.flatMap { $0.innerState() }
                + slots.flatMap { $0.cache.flatMap { $0.innerState() } })
        batch.commit(advancedBy: width)
        let finishing = slots.filter { completedForward(for: $0) }
        if !finishing.isEmpty {
            Stream().synchronize()
            for slot in finishing { settle(slot) }
        }

        batchedForwardCount += 1
        maximumBatchSize = max(maximumBatchSize, slots.count)
        for (row, slot) in slots.enumerated() where !slot.finishing {
            do {
                var rowState = output.state ?? state
                if let hidden = output.state?[mtpLastHiddenStatesKey] {
                    rowState[mtpLastHiddenStatesKey] = hidden[row ..< row + 1]
                }
                rowState[mtpSharedKVOffsetsKey] = [
                    "full_attention": prepared[row].targetOffsetAfterVerification
                ]
                var iterator = slot.iterator!
                iterator.completeScheduledVerification(
                    prepared[row],
                    result: LMOutput(
                        logits: output.logits[row ..< row + 1], state: rowState),
                    cacheAlreadyEvaluated: true)
                slot.iterator = iterator
                slot.cache = iterator.mainCache
                guard let token = slot.iterator!.next() else {
                    finish(slot, reason: .length)
                    continue
                }
                let arrays = slot.iterator!.scheduledResidentArrays
                eval(arrays)
                guard arrays.reduce(0, { $0 + $1.nbytes }) <= slot.reservation else {
                    throw ConcurrentTextRuntimeError.memoryBudgetExceeded
                }
                try accept(token, slot: slot)
            } catch { finish(slot, error: error) }
        }
    }

    private func batchStep(_ slots: [Slot]) async throws {
        let batch = try RuntimeBatchCache(rows: slots.map(\.cache))
        beginForward(for: slots)
        let logits: MLXArray
        do {
            logits = try await model.scheduledForward(
                MLXArray(slots.map { $0.lastToken! }).expandedDimensions(axis: 1),
                cache: batch.cache)
        } catch {
            let finishing = slots.filter { completedForward(for: $0) }
            if !finishing.isEmpty {
                Stream().synchronize()
                for slot in finishing { settle(slot) }
            }
            throw error
        }
        eval(
            [logits] + batch.cache.flatMap { $0.innerState() }
                + slots.flatMap { $0.cache.flatMap { $0.innerState() } })
        batch.commit()
        let finishing = slots.filter { completedForward(for: $0) }
        if !finishing.isEmpty {
            Stream().synchronize()
            for slot in finishing { settle(slot) }
        }
        batchedForwardCount += 1
        maximumBatchSize = max(maximumBatchSize, slots.count)
        for (row, slot) in slots.enumerated() {
            guard !slot.finishing else { continue }
            do {
                slot.lookupStorage?.commitProcessedTokens(1)
                try compress(slot)
                let bytes = slot.cache.flatMap { $0.innerState() }.reduce(0) { $0 + $1.nbytes }
                guard bytes <= slot.reservation - configuration.workingMemoryBytes else {
                    throw ConcurrentTextRuntimeError.memoryBudgetExceeded
                }
                try sample(logits[row ..< row + 1], slot: slot)
            } catch { finish(slot, error: error) }
        }
    }

    private func checkpoint(_ slot: Slot, bytes: Int) {
        guard model.scheduledSupportsPrefixCache else { return }
        let tokens = Array(slot.request.tokens.prefix(slot.position))
        let bytes = bytes + tokens.count * MemoryLayout<Int>.stride
        do { try persistentStore?.store(tokens: tokens, cache: slot.cache) } catch {
            persistentCacheFailures += 1
        }
        guard bytes <= configuration.prefixCacheBytes,
            !prefixes.contains(where: { !$0.speculative && $0.tokens == tokens })
        else { return }
        while prefixBytes > configuration.prefixCacheBytes - bytes { prefixes.removeFirst() }
        let cache = slot.cache.map { $0.copy() }
        eval(cache.flatMap { $0.innerState() })
        prefixes.append(Prefix(tokens: tokens, state: .ordinary(cache), bytes: bytes))
    }

    private func checkpointSpeculative(_ slot: Slot) {
        guard model.scheduledSupportsPrefixCache else { return }
        guard let iterator = slot.iterator, let cacheBytes = iterator.scheduledPrefixCacheBytes
        else {
            return
        }
        // The shifted drafter cache includes the next prompt token, unlike the target cache.
        let count = slot.position + 1
        let (tokenBytes, tokenOverflow) = count.multipliedReportingOverflow(
            by: MemoryLayout<Int>.stride)
        let (bytes, overflow) = cacheBytes.addingReportingOverflow(tokenBytes)
        guard !tokenOverflow, !overflow else { return }
        let tokens = Array(slot.request.tokens.prefix(count))
        guard let snapshot = iterator.snapshotScheduledPrefix() else { return }
        do {
            try persistentSpeculativeStore?.store(
                tokens: tokens, cache: snapshot.persistentCaches,
                processedTokenCount: snapshot.processedTokenCount)
        } catch { persistentCacheFailures += 1 }
        guard bytes <= configuration.prefixCacheBytes,
            !prefixes.contains(where: { $0.speculative && $0.tokens == tokens })
        else { return }
        while prefixBytes > configuration.prefixCacheBytes - bytes { prefixes.removeFirst() }
        prefixes.append(Prefix(tokens: tokens, state: .speculative(snapshot), bytes: bytes))
    }

    private func emit(_ event: Event, to slot: Slot) throws {
        switch slot.continuation.yield(event) {
        case .enqueued: break
        case .dropped: throw ConcurrentTextRuntimeError.consumerTooSlow
        case .terminated: throw CancellationError()
        @unknown default: throw ConcurrentTextRuntimeError.consumerTooSlow
        }
    }

    private func finish(_ slot: Slot, reason: GenerateStopReason? = nil, error: Error? = nil) {
        guard !slot.finished else { return }
        if !slot.finishing {
            slot.finishing = true
            slot.finishReason = reason
            slot.finishError = error
            if case .cancelled? = reason { slot.cancellationRequested = true }
        }
        guard slot.inFlightForwardCount == 0 else { return }
        settle(slot)
    }

    private func settle(_ slot: Slot) {
        guard slot.finishing, !slot.finished, slot.inFlightForwardCount == 0 else { return }
        slot.finished = true
        releasedReservations.append(slot.id)
        if slot.modelRequestBegan {
            model.scheduledEndRequest()
            slot.modelRequestBegan = false
        }
        active.removeAll { $0.id == slot.id }
        pending.removeAll { $0.id == slot.id }
        if let storage = slot.lookupStorage {
            let consumed = min(
                slot.lookupPendingIndex, slot.lookupCommittedPendingTokenCount)
            let lookahead = slot.lookupCommittedPendingTokenCount - consumed
            if lookahead > 0 {
                if slot.lookupNativeHybridRound {
                    storage.rewindSpeculative(lookahead)
                } else {
                    storage.rewindLastRound(lookahead)
                }
            }
        }
        discardSpeculativePromptCacheCheckpoints(slot.cache)
        var error = slot.finishError
        let telemetry =
            slot.iterator?.speculativeDecodingTelemetry
            ?? (slot.lookupTelemetry.roundCount > 0 ? slot.lookupTelemetry : nil)
        if error == nil, let telemetry {
            if case .dropped = slot.continuation.yield(.speculation(telemetry)) {
                error = ConcurrentTextRuntimeError.consumerTooSlow
            }
        }
        slot.iterator?.finalizeGeneration()
        if let iterator = slot.iterator { eval(iterator.scheduledResidentArrays) }
        slot.iterator = nil
        slot.lookupStorage = nil
        slot.lookup = nil
        slot.cache.removeAll()
        if let error {
            slot.continuation.finish(throwing: error)
        } else {
            if let reason = slot.finishReason,
                case .dropped = slot.continuation.yield(.finished(reason))
            {
                slot.continuation.finish(throwing: ConcurrentTextRuntimeError.consumerTooSlow)
            } else {
                slot.continuation.finish()
            }
        }
        let waiters = slot.settlementWaiters
        slot.settlementWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }
}
