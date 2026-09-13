// Copyright © 2026 Faux Fox.

import Foundation

/// A loaded expert and the exact number of bytes retained for it.
///
/// MLX arrays are not `Sendable`. Wrap an immutable expert payload here only after
/// arranging that all use of its contained values remains in the owning inference
/// isolation domain.
public struct ExpertWeight<Value>: @unchecked Sendable {
    /// The bytes retained by the payload, including every discontiguous tensor.
    public let byteCount: Int
    private let value: Value

    /// Creates an opaque expert payload with its retained byte count.
    public init(unchecked value: Value, byteCount: Int) throws {
        guard byteCount >= 0 else { throw ExpertWeightStoreError.invalidByteCount }
        self.value = value
        self.byteCount = byteCount
    }

    /// Uses the payload without exposing it as a `Sendable` value.
    public func withValue<Result>(_ body: (Value) throws -> Result) rethrows -> Result {
        try body(value)
    }
}

/// Errors reported by ``ExpertWeightStore`` and ``ExpertWeightLRUCache``.
public enum ExpertWeightStoreError: Error, Sendable, Equatable {
    /// A cache capacity must not be negative.
    case invalidCapacity
    /// An expert's retained byte count must not be negative.
    case invalidByteCount
}

/// An exact, bounded least-recently-used cache for hot expert weights.
///
/// This type is deliberately synchronous because its values can contain non-Sendable
/// MLX arrays. Keep it in one isolation domain, or use ``ExpertWeightStore``.
public struct ExpertWeightLRUCache<Key: Hashable & Sendable, Value> {
    /// An entry evicted to make room for a more recently used expert.
    public struct Eviction: Sendable, Equatable {
        /// The evicted expert's key.
        public let key: Key
        /// The exact number of bytes released by the evicted entry.
        public let byteCount: Int
    }

    /// The result of inserting an expert.
    public enum Insertion: Sendable, Equatable {
        /// The expert was retained. Associated entries were evicted first.
        case stored([Eviction])
        /// The expert is larger than the cache and was returned to the caller uncached.
        case notStoredOversized
    }

    private struct Entry {
        var value: Value
        let byteCount: Int
        var recency: UInt64
    }

    /// Maximum number of bytes that may be retained.
    public private(set) var capacityBytes: Int
    /// Exact bytes retained by cache entries.
    public private(set) var residentBytes = 0
    /// The number of retained experts.
    public var count: Int { entries.count }

    private var entries: [Key: Entry] = [:]
    private var nextRecency: UInt64 = 0

    /// Creates an empty cache with a fixed byte capacity.
    public init(capacityBytes: Int) throws {
        guard capacityBytes >= 0 else { throw ExpertWeightStoreError.invalidCapacity }
        self.capacityBytes = capacityBytes
    }

    /// Returns a hot expert and marks it as most recently used.
    public mutating func value(for key: Key) -> Value? {
        guard var entry = entries[key] else { return nil }
        entry.recency = advanceRecency()
        entries[key] = entry
        return entry.value
    }

    /// Returns an expert without changing its LRU position.
    public func peekValue(for key: Key) -> Value? {
        entries[key]?.value
    }

    /// Inserts an expert, evicting least-recently-used entries as necessary.
    ///
    /// An oversized expert never evicts existing entries. The cache remains unchanged
    /// and returns `.notStoredOversized` so a streaming caller may still use it once.
    @discardableResult
    public mutating func insert(
        _ value: Value, for key: Key, byteCount: Int
    ) throws -> Insertion {
        guard byteCount >= 0 else { throw ExpertWeightStoreError.invalidByteCount }
        guard byteCount <= capacityBytes else { return .notStoredOversized }

        let existing = entries.removeValue(forKey: key)
        if let existing {
            residentBytes -= existing.byteCount
        }

        var evictions: [Eviction] = []
        while residentBytes > capacityBytes - byteCount {
            guard let victim = leastRecentlyUsedKey() else { break }
            let entry = entries.removeValue(forKey: victim)!
            residentBytes -= entry.byteCount
            evictions.append(Eviction(key: victim, byteCount: entry.byteCount))
        }

        entries[key] = Entry(value: value, byteCount: byteCount, recency: advanceRecency())
        residentBytes += byteCount
        return .stored(evictions)
    }

    /// Removes one retained expert and returns its payload.
    @discardableResult
    public mutating func removeValue(for key: Key) -> Value? {
        guard let entry = entries.removeValue(forKey: key) else { return nil }
        residentBytes -= entry.byteCount
        return entry.value
    }

    /// Shrinks or expands the capacity and evicts least-recently-used entries if needed.
    @discardableResult
    public mutating func setCapacityBytes(_ capacityBytes: Int) throws -> [Eviction] {
        guard capacityBytes >= 0 else { throw ExpertWeightStoreError.invalidCapacity }
        self.capacityBytes = capacityBytes
        var evictions: [Eviction] = []
        while residentBytes > capacityBytes {
            guard let victim = leastRecentlyUsedKey() else { break }
            let entry = entries.removeValue(forKey: victim)!
            residentBytes -= entry.byteCount
            evictions.append(Eviction(key: victim, byteCount: entry.byteCount))
        }
        return evictions
    }

    /// Removes every retained expert and returns the bytes released.
    @discardableResult
    public mutating func removeAll() -> Int {
        let releasedBytes = residentBytes
        entries.removeAll(keepingCapacity: false)
        residentBytes = 0
        return releasedBytes
    }

    private mutating func advanceRecency() -> UInt64 {
        if nextRecency == .max {
            normalizeRecency()
        }
        nextRecency += 1
        return nextRecency
    }

    private mutating func normalizeRecency() {
        let ordered = entries.sorted { lhs, rhs in
            if lhs.value.recency != rhs.value.recency {
                return lhs.value.recency < rhs.value.recency
            }
            return String(reflecting: lhs.key) < String(reflecting: rhs.key)
        }
        for (index, pair) in ordered.enumerated() {
            entries[pair.key]?.recency = UInt64(index)
        }
        nextRecency = UInt64(ordered.count)
    }

    private func leastRecentlyUsedKey() -> Key? {
        entries.min { lhs, rhs in
            if lhs.value.recency != rhs.value.recency {
                return lhs.value.recency < rhs.value.recency
            }
            return String(reflecting: lhs.key) < String(reflecting: rhs.key)
        }?.key
    }
}

/// An actor-owned streamed-expert store with load coalescing and LRU retention.
///
/// The store only owns decoded payloads and byte accounting. Its loader can obtain a
/// quantized expert from any backend, including a manifest with many independent ranges.
public actor ExpertWeightStore<Key: Hashable & Sendable, Value> {
    /// Loads one opaque expert payload for a key.
    public typealias Loader = @Sendable (Key) async throws -> ExpertWeight<Value>

    /// A snapshot suitable for assertions, diagnostics, and memory-pressure decisions.
    public struct Status: Sendable {
        /// The cache's hard byte ceiling.
        public let capacityBytes: Int
        /// Exact bytes retained by cached experts.
        public let residentBytes: Int
        /// Number of cached experts.
        public let cachedExpertCount: Int
        /// Number of unique loads currently in progress.
        public let inFlightLoadCount: Int
    }

    private struct Flight {
        let id: UUID
        let task: Task<ExpertWeight<Value>, Error>
        var keepAlive: Bool
        var waiters: [UUID: CheckedContinuation<ExpertWeight<Value>, Error>]
    }

    private let loader: Loader
    private var cache: ExpertWeightLRUCache<Key, ExpertWeight<Value>>
    private var flights: [Key: Flight] = [:]

    /// Creates a store with a strict cache capacity and an abstract loading backend.
    public init(capacityBytes: Int, loader: @escaping Loader) throws {
        cache = try ExpertWeightLRUCache(capacityBytes: capacityBytes)
        self.loader = loader
    }

    /// Returns a cached expert or coalesces with one asynchronous load for its key.
    ///
    /// Cancelling the last direct waiter cancels that load. A preloaded expert remains
    /// alive until it completes or is explicitly cancelled.
    public func expert(for key: Key) async throws -> ExpertWeight<Value> {
        if let expert = cache.value(for: key) { return expert }

        let flightID = startFlight(for: key, keepAlive: false)
        let waiterID = UUID()
        return try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { continuation in
                guard var flight = flights[key], flight.id == flightID else {
                    continuation.resume(throwing: CancellationError())
                    return
                }
                flight.waiters[waiterID] = continuation
                flights[key] = flight
            }
        } onCancel: {
            Task { await self.cancelWaiter(for: key, flightID: flightID, waiterID: waiterID) }
        }
    }

    /// Starts background loads for likely next experts without retaining duplicate work.
    public func preload<S: Sequence>(_ keys: S) where S.Element == Key {
        for key in keys {
            guard cache.peekValue(for: key) == nil else { continue }
            _ = startFlight(for: key, keepAlive: true)
        }
    }

    /// Cancels all in-flight preload work that has no direct waiter.
    public func cancelPreloads() {
        let keys = Array(flights.keys)
        for key in keys {
            guard var flight = flights[key], flight.keepAlive else { continue }
            flight.keepAlive = false
            if flight.waiters.isEmpty {
                flights[key] = nil
                flight.task.cancel()
            } else {
                flights[key] = flight
            }
        }
    }

    /// Cancels all pending loads and releases all cached experts for memory pressure.
    public func handleMemoryWarning() {
        _ = cache.removeAll()
        cancelAllLoads()
    }

    /// Cancels all pending loads while retaining hot cached experts.
    public func cancelAllLoads() {
        let activeFlights = flights
        flights.removeAll()
        for flight in activeFlights.values {
            flight.task.cancel()
            for waiter in flight.waiters.values {
                waiter.resume(throwing: CancellationError())
            }
        }
    }

    /// Removes one cached expert. The current load, if any, is left running.
    public func removeCachedExpert(for key: Key) {
        _ = cache.removeValue(for: key)
    }

    /// Adjusts the cache ceiling, evicting least-recently-used experts immediately.
    public func setCapacityBytes(_ capacityBytes: Int) throws {
        _ = try cache.setCapacityBytes(capacityBytes)
    }

    /// Returns exact cache accounting and the current number of unique loads.
    public func status() -> Status {
        Status(
            capacityBytes: cache.capacityBytes, residentBytes: cache.residentBytes,
            cachedExpertCount: cache.count, inFlightLoadCount: flights.count)
    }

    private func startFlight(for key: Key, keepAlive: Bool) -> UUID {
        if var existing = flights[key] {
            existing.keepAlive = existing.keepAlive || keepAlive
            flights[key] = existing
            return existing.id
        }

        let id = UUID()
        let loader = loader
        let task = Task.detached { try await loader(key) }
        flights[key] = Flight(id: id, task: task, keepAlive: keepAlive, waiters: [:])

        Task { [weak self] in
            let result: Result<ExpertWeight<Value>, Error>
            do {
                result = .success(try await task.value)
            } catch {
                result = .failure(error)
            }
            await self?.finishFlight(for: key, id: id, result: result)
        }
        return id
    }

    private func cancelWaiter(for key: Key, flightID: UUID, waiterID: UUID) {
        guard var flight = flights[key], flight.id == flightID,
            let waiter = flight.waiters.removeValue(forKey: waiterID)
        else { return }

        waiter.resume(throwing: CancellationError())
        if flight.waiters.isEmpty && !flight.keepAlive {
            flights[key] = nil
            flight.task.cancel()
        } else {
            flights[key] = flight
        }
    }

    private func finishFlight(
        for key: Key, id: UUID, result: Result<ExpertWeight<Value>, Error>
    ) {
        guard let flight = flights[key], flight.id == id else { return }
        flights[key] = nil

        switch result {
        case .success(let expert):
            _ = try? cache.insert(expert, for: key, byteCount: expert.byteCount)
            for waiter in flight.waiters.values {
                waiter.resume(returning: expert)
            }
        case .failure(let error):
            for waiter in flight.waiters.values {
                waiter.resume(throwing: error)
            }
        }
    }
}
