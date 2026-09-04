// Copyright © 2026 Faux Fox.

import Foundation

/// Shared admission accounting for every model in one native inference service.
public actor InferenceResourceBudget {
    public struct Status: Sendable {
        public let capacityBytes: Int
        public let residentBytes: Int
        public let requestBytes: Int
        public let interactiveHeadroomBytes: Int
        public let pendingExecutionTurns: Int
    }

    public let capacityBytes: Int
    public let interactiveHeadroomBytes: Int
    private var residents: [UUID: Int] = [:]
    private var requests: [UUID: Int] = [:]
    private var demands: [(id: UUID, bytes: Int, background: Bool)] = []
    private var revision: UInt64 = 0
    private var waiters: [CheckedContinuation<Void, Never>] = []
    private struct Turn {
        let id: UUID
        let background: Bool
        let continuation: CheckedContinuation<Bool, Never>
    }
    private var turnOwner: (id: UUID, background: Bool)?
    private var turnWaiters: [Turn] = []
    private var turnService: [UUID: Double] = [:]

    public init(capacityBytes: Int, interactiveHeadroomBytes: Int) throws {
        guard capacityBytes > 0, interactiveHeadroomBytes >= 0,
            interactiveHeadroomBytes < capacityBytes
        else {
            throw ConcurrentTextRuntimeError.invalidConfiguration
        }
        self.capacityBytes = capacityBytes
        self.interactiveHeadroomBytes = interactiveHeadroomBytes
    }

    public func status() -> Status {
        Status(
            capacityBytes: capacityBytes, residentBytes: residentBytes,
            requestBytes: requestBytes, interactiveHeadroomBytes: interactiveHeadroomBytes,
            pendingExecutionTurns: turnWaiters.count)
    }

    private var residentBytes: Int { residents.values.reduce(0, +) }
    private var requestBytes: Int { requests.values.reduce(0, +) }

    func register(_ owner: UUID, bytes: Int) throws {
        let previous = residents[owner] ?? 0
        guard bytes >= 0, bytes <= capacityBytes - residentBytes - requestBytes + previous else {
            throw ConcurrentTextRuntimeError.memoryBudgetExceeded
        }
        residents[owner] = bytes
        signal()
    }

    func unregister(_ owner: UUID) {
        residents[owner] = nil
        signal()
    }

    func canEverFit(_ bytes: Int, background: Bool) -> Bool {
        bytes <= capacityBytes - residentBytes - (background ? interactiveHeadroomBytes : 0)
    }

    func acquire(_ id: UUID, bytes: Int, background: Bool) throws -> Bool {
        guard canEverFit(bytes, background: background) else {
            throw ConcurrentTextRuntimeError.memoryBudgetExceeded
        }
        if requests[id] != nil { return true }
        if !demands.contains(where: { $0.id == id }) {
            demands.append((id, bytes, background))
        }
        let available = capacityBytes - residentBytes - requestBytes
        let first = demands.first!
        let firstFits = first.bytes <= available - (first.background ? interactiveHeadroomBytes : 0)
        // Interactive work can use its reserved headroom while background admission waits.
        let eligible = firstFits ? first.id : demands.first(where: { !$0.background })?.id
        guard eligible == id,
            bytes <= available - (background ? interactiveHeadroomBytes : 0)
        else { return false }
        demands.removeAll { $0.id == id }
        requests[id] = bytes
        signal()
        return true
    }

    func release(_ id: UUID) {
        let cancelled = turnWaiters.filter { $0.id == id }
        turnWaiters.removeAll { $0.id == id }
        for waiter in cancelled { waiter.continuation.resume(returning: false) }
        if turnOwner?.id == id { endTurn(id) }
        turnService[id] = nil
        demands.removeAll { $0.id == id }
        requests[id] = nil
        signal()
    }

    /// Serializes bounded GPU turns across models under the same weighted-service policy.
    func beginTurn(_ id: UUID, background: Bool) async -> Bool {
        guard requests[id] != nil else { return false }
        if turnService[id] == nil { turnService[id] = turnService.values.min() ?? 0 }
        if turnOwner == nil {
            turnOwner = (id, background)
            return true
        }
        return await withCheckedContinuation {
            turnWaiters.append(Turn(id: id, background: background, continuation: $0))
            signal()
        }
    }

    func endTurn(_ id: UUID) {
        guard let owner = turnOwner, owner.id == id else { return }
        turnService[id, default: 0] += owner.background ? 3 : 1
        turnOwner = nil
        if let index = turnWaiters.indices.min(by: {
            turnService[turnWaiters[$0].id, default: 0]
                < turnService[turnWaiters[$1].id, default: 0]
        }) {
            let next = turnWaiters.remove(at: index)
            turnOwner = (next.id, next.background)
            next.continuation.resume(returning: true)
        }
    }

    func version() -> UInt64 { revision }

    func waitForChange(after observed: UInt64) async {
        if observed != revision { return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func signal() {
        revision &+= 1
        let pending = waiters
        waiters.removeAll()
        for waiter in pending { waiter.resume() }
    }
}

/// Embeddable model lifecycle and routing, independent of a transport or conversation.
public actor NativeInferenceRuntime {
    public let resources: InferenceResourceBudget
    private var models: [String: ConcurrentTextRuntime] = [:]
    private var loading: Set<String> = []
    private var cancelledLoads: Set<String> = []
    private var unloadWaiters: [String: [CheckedContinuation<Void, Never>]] = [:]

    public init(memoryBudgetBytes: Int, interactiveHeadroomBytes: Int) throws {
        resources = try InferenceResourceBudget(
            capacityBytes: memoryBudgetBytes, interactiveHeadroomBytes: interactiveHeadroomBytes)
    }

    /// Reserve resident/loading memory before calling the loader. Include load-time workspace.
    /// The loader must not retain another owner of the returned runtime's model.
    public func load(
        id: String, estimatedResidentBytes: Int,
        loader: @Sendable () async throws -> ConcurrentTextRuntime
    ) async throws {
        guard models[id] == nil, !loading.contains(id) else {
            throw ConcurrentTextRuntimeError.invalidConfiguration
        }
        let reservation = UUID()
        loading.insert(id)
        defer { finishLoading(id) }
        do {
            try await resources.register(reservation, bytes: estimatedResidentBytes)
            guard !cancelledLoads.contains(id) else { throw CancellationError() }
            let runtime = try await loader()
            do {
                try Task.checkCancellation()
                guard !cancelledLoads.contains(id) else { throw CancellationError() }
                try await runtime.attach(
                    resources: resources, reservation: reservation,
                    maximumResidentBytes: estimatedResidentBytes)
                try Task.checkCancellation()
                guard !cancelledLoads.contains(id) else { throw CancellationError() }
                models[id] = runtime
            } catch {
                await runtime.shutdown()
                throw error
            }
        } catch {
            await resources.unregister(reservation)
            throw error
        }
    }

    private func finishLoading(_ id: String) {
        loading.remove(id)
        cancelledLoads.remove(id)
        let waiters = unloadWaiters.removeValue(forKey: id) ?? []
        for waiter in waiters { waiter.resume() }
    }

    public func modelIDs() -> [String] { models.keys.sorted() }

    public func capabilities(modelID: String) throws -> ConcurrentTextRuntime.Capabilities {
        guard let model = models[modelID] else { throw ConcurrentTextRuntimeError.modelNotLoaded }
        return model.capabilities
    }

    public func generate(modelID: String, request: ConcurrentTextRuntime.Request) async throws
        -> ConcurrentTextRuntime.Generation
    {
        guard let model = models[modelID] else { throw ConcurrentTextRuntimeError.modelNotLoaded }
        return try await model.generate(request)
    }

    public func cancel(modelID: String, requestID: UUID) async {
        await models[modelID]?.cancel(requestID)
    }

    public func clearCaches(modelID: String, persistent: Bool = false) async throws {
        guard let model = models[modelID] else { throw ConcurrentTextRuntimeError.modelNotLoaded }
        try await model.clearCaches(persistent: persistent)
    }

    public func unload(modelID: String) async {
        if loading.contains(modelID) {
            cancelledLoads.insert(modelID)
            await withCheckedContinuation { unloadWaiters[modelID, default: []].append($0) }
            return
        }
        guard let runtime = models.removeValue(forKey: modelID) else { return }
        await runtime.shutdown()
    }
}
