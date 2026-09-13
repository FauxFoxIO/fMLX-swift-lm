// Copyright © 2026 Faux Fox.

import XCTest

@testable import MLXLMCommon

final class ExpertWeightStoreTests: XCTestCase {
    func testCacheEvictsLeastRecentlyUsedAndAccountsExactly() throws {
        var cache = try ExpertWeightLRUCache<String, String>(capacityBytes: 10)
        XCTAssertEqual(try cache.insert("a", for: "a", byteCount: 3), .stored([]))
        XCTAssertEqual(try cache.insert("b", for: "b", byteCount: 4), .stored([]))
        XCTAssertEqual(cache.residentBytes, 7)

        XCTAssertEqual(cache.value(for: "a"), "a")
        XCTAssertEqual(
            try cache.insert("c", for: "c", byteCount: 5),
            .stored([.init(key: "b", byteCount: 4)]))

        XCTAssertEqual(cache.residentBytes, 8)
        XCTAssertEqual(cache.count, 2)
        XCTAssertEqual(cache.peekValue(for: "a"), "a")
        XCTAssertNil(cache.peekValue(for: "b"))
        XCTAssertEqual(cache.peekValue(for: "c"), "c")

        XCTAssertEqual(try cache.insert("oversized", for: "d", byteCount: 11), .notStoredOversized)
        XCTAssertEqual(cache.residentBytes, 8)
        XCTAssertEqual(cache.count, 2)
    }

    func testCacheReplacementEvictsBeforeExceedingCapacity() throws {
        var cache = try ExpertWeightLRUCache<String, Int>(capacityBytes: 10)
        _ = try cache.insert(1, for: "a", byteCount: 4)
        _ = try cache.insert(2, for: "b", byteCount: 4)

        XCTAssertEqual(
            try cache.insert(3, for: "a", byteCount: 7),
            .stored([.init(key: "b", byteCount: 4)]))
        XCTAssertEqual(cache.residentBytes, 7)
        XCTAssertEqual(cache.count, 1)
        XCTAssertEqual(cache.value(for: "a"), 3)
        XCTAssertNil(cache.value(for: "b"))
    }

    func testStoreCoalescesLoadersAndCachesTheWholeExpertByteCount() async throws {
        let gate = ExpertLoadGate()
        let store = try ExpertWeightStore<Int, [Int]>(capacityBytes: 8) { key in
            await gate.recordStart(for: key)
            await gate.waitForRelease()
            try Task.checkCancellation()
            return try ExpertWeight(unchecked: [key], byteCount: 4)
        }

        let first = Task { try await store.expert(for: 7) }
        let second = Task { try await store.expert(for: 7) }
        await gate.waitUntilStarted(for: 7)
        let startCount = await gate.startCount(for: 7)
        XCTAssertEqual(startCount, 1)

        await gate.release()
        let firstValue = try await first.value
        let secondValue = try await second.value
        XCTAssertEqual(firstValue.withValue { $0 }, [7])
        XCTAssertEqual(secondValue.withValue { $0 }, [7])

        let status = await store.status()
        XCTAssertEqual(status.residentBytes, 4)
        XCTAssertEqual(status.cachedExpertCount, 1)
        XCTAssertEqual(status.inFlightLoadCount, 0)
    }

    func testCancellingLastWaiterCancelsAnUncachedLoad() async throws {
        let gate = ExpertLoadGate()
        let store = try ExpertWeightStore<Int, Int>(capacityBytes: 4) { key in
            await gate.recordStart(for: key)
            await gate.waitForRelease()
            try Task.checkCancellation()
            return try ExpertWeight(unchecked: key, byteCount: 4)
        }

        let task = Task { try await store.expert(for: 1) }
        await gate.waitUntilStarted(for: 1)
        task.cancel()

        do {
            _ = try await task.value
            XCTFail("cancelled waiters must not receive an expert")
        } catch is CancellationError {
        } catch {
            XCTFail("expected CancellationError, got \(error)")
        }

        let status = await store.status()
        XCTAssertEqual(status.inFlightLoadCount, 0)
        XCTAssertEqual(status.cachedExpertCount, 0)
        await gate.release()
    }

    func testMemoryWarningDropsCachedExpertsAndCancelsPreloads() async throws {
        let gate = ExpertLoadGate()
        let store = try ExpertWeightStore<Int, Int>(capacityBytes: 8) { key in
            if key == 3 {
                await gate.recordStart(for: key)
                await gate.waitForRelease()
                try Task.checkCancellation()
            }
            return try ExpertWeight(unchecked: key, byteCount: 4)
        }

        _ = try await store.expert(for: 1)
        _ = try await store.expert(for: 2)
        await store.preload([3])
        await gate.waitUntilStarted(for: 3)
        await store.handleMemoryWarning()

        let status = await store.status()
        XCTAssertEqual(status.residentBytes, 0)
        XCTAssertEqual(status.cachedExpertCount, 0)
        XCTAssertEqual(status.inFlightLoadCount, 0)
        await gate.release()
    }
}

private actor ExpertLoadGate {
    private var starts: [Int: Int] = [:]
    private var startWaiters: [Int: [CheckedContinuation<Void, Never>]] = [:]
    private var released = false
    private var releaseWaiters: [CheckedContinuation<Void, Never>] = []

    func recordStart(for key: Int) {
        starts[key, default: 0] += 1
        let waiters = startWaiters.removeValue(forKey: key) ?? []
        for waiter in waiters { waiter.resume() }
    }

    func startCount(for key: Int) -> Int { starts[key, default: 0] }

    func waitUntilStarted(for key: Int) async {
        guard starts[key, default: 0] == 0 else { return }
        await withCheckedContinuation { continuation in
            startWaiters[key, default: []].append(continuation)
        }
    }

    func waitForRelease() async {
        guard !released else { return }
        await withCheckedContinuation { releaseWaiters.append($0) }
    }

    func release() {
        released = true
        let waiters = releaseWaiters
        releaseWaiters.removeAll()
        for waiter in waiters { waiter.resume() }
    }
}
