// Copyright © 2026 Faux Fox.

import Foundation
import MLXLMCommon
import XCTest

@testable import MLXLLM

final class Qwen35StreamedExpertExecutionTests: XCTestCase {
    func testStreamedModelDoesNotRegisterResidentExpertParameters() throws {
        let configuration = try tinyConfiguration()
        let resident = Qwen35TextModel(configuration)
        let streamed = Qwen35TextModel(configuration, streamedExperts: true)

        XCTAssertTrue(
            resident.parameters().flattened().contains { $0.0.contains(".switch_mlp.") })
        XCTAssertFalse(
            streamed.parameters().flattened().contains { $0.0.contains(".switch_mlp.") })
    }

    func testCompactsUniqueExpertsAndPreservesRouterOrder() async throws {
        let loads = ExpertLoadRecorder()
        let store = try ExpertWeightStore<Int, Int>(capacityBytes: 64) { expertID in
            await loads.record(expertID)
            return try ExpertWeight(unchecked: expertID, byteCount: 4)
        }
        let execution = Qwen35StreamedExpertExecution(store: store)

        let plan = try await execution.prepareDecode(
            sequenceLength: 2,
            topK: 3,
            expertCount: 8,
            routerExpertIDs: [7, 2, 7, 5, 2, 1])

        XCTAssertEqual(plan.expertIDs, [7, 2, 5, 1])
        XCTAssertEqual(plan.compactRouterIDs, [0, 1, 0, 2, 1, 3])
        XCTAssertEqual(plan.compactExpertCount, 4)
        XCTAssertEqual(plan.withExpertWeights { $0.map { $0.withValue { $0 } } }, [7, 2, 5, 1])
        let loadedIDs = await loads.values
        XCTAssertEqual(loadedIDs, [7, 2, 5, 1])
    }

    func testRejectsPrefillWithoutLoadingAnyExperts() async throws {
        let loads = ExpertLoadRecorder()
        let store = try ExpertWeightStore<Int, Int>(capacityBytes: 64) { expertID in
            await loads.record(expertID)
            return try ExpertWeight(unchecked: expertID, byteCount: 4)
        }
        let execution = Qwen35StreamedExpertExecution(store: store)

        do {
            _ = try await execution.prepareDecode(
                sequenceLength: 3, topK: 1, expertCount: 8, routerExpertIDs: [0, 1, 2])
            XCTFail("prefill must remain resident")
        } catch let error as Qwen35StreamedExpertExecutionError {
            XCTAssertEqual(error, .requiresResidentExperts(sequenceLength: 3))
        }
        let loadedIDs = await loads.values
        XCTAssertEqual(loadedIDs, [])
    }

    func testRejectsZeroTopKAndOutOfRangeAssignments() async throws {
        let store = try ExpertWeightStore<Int, Int>(capacityBytes: 64) { expertID in
            try ExpertWeight(unchecked: expertID, byteCount: 4)
        }
        let execution = Qwen35StreamedExpertExecution(store: store)

        await assertPreparationError(
            execution,
            sequenceLength: 1, topK: 0, expertCount: 8, routerExpertIDs: [],
            expected: .invalidTopK(topK: 0, expertCount: 8))
        await assertPreparationError(
            execution,
            sequenceLength: 1, topK: 1, expertCount: 8, routerExpertIDs: [8],
            expected: .invalidRouterExpertID(8))
    }

    func testSerializesPreparationUntilTheCurrentLoadsFinish() async throws {
        let gate = ExpertLoadGate()
        let store = try ExpertWeightStore<Int, Int>(capacityBytes: 64) { expertID in
            await gate.recordStart(expertID)
            if expertID == 1 {
                await gate.waitForRelease()
            }
            return try ExpertWeight(unchecked: expertID, byteCount: 4)
        }
        let execution = Qwen35StreamedExpertExecution(store: store)

        let first = Task {
            try await execution.prepareDecode(
                sequenceLength: 1, topK: 1, expertCount: 8, routerExpertIDs: [1])
        }
        await gate.waitUntilStarted(1)
        let second = Task {
            try await execution.prepareDecode(
                sequenceLength: 1, topK: 1, expertCount: 8, routerExpertIDs: [2])
        }
        await Task.yield()
        let initiallyStartedIDs = await gate.startedIDs
        XCTAssertEqual(initiallyStartedIDs, [1])

        await gate.release()
        _ = try await first.value
        _ = try await second.value
        let startedIDs = await gate.startedIDs
        XCTAssertEqual(startedIDs, [1, 2])
    }

    private func assertPreparationError(
        _ execution: Qwen35StreamedExpertExecution<Int>,
        sequenceLength: Int,
        topK: Int,
        expertCount: Int,
        routerExpertIDs: [Int],
        expected: Qwen35StreamedExpertExecutionError,
        file: StaticString = #filePath,
        line: UInt = #line
    ) async {
        do {
            _ = try await execution.prepareDecode(
                sequenceLength: sequenceLength,
                topK: topK,
                expertCount: expertCount,
                routerExpertIDs: routerExpertIDs)
            XCTFail("expected \(expected)", file: file, line: line)
        } catch let error as Qwen35StreamedExpertExecutionError {
            XCTAssertEqual(error, expected, file: file, line: line)
        } catch {
            XCTFail("unexpected error \(error)", file: file, line: line)
        }
    }

    private func tinyConfiguration() throws -> Qwen35TextConfiguration {
        let json = """
            {
                "model_type": "qwen3_5_moe",
                "hidden_size": 16,
                "num_hidden_layers": 1,
                "intermediate_size": 32,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 8,
                "linear_num_value_heads": 2,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 8,
                "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 4,
                "vocab_size": 32,
                "full_attention_interval": 1,
                "num_experts": 4,
                "num_experts_per_tok": 2,
                "moe_intermediate_size": 16,
                "shared_expert_intermediate_size": 16
            }
            """
        return try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: Data(json.utf8))
    }
}

private actor ExpertLoadRecorder {
    private(set) var values: [Int] = []

    func record(_ value: Int) {
        values.append(value)
    }
}

private actor ExpertLoadGate {
    private var released = false
    private var releaseWaiters: [CheckedContinuation<Void, Never>] = []
    private var startWaiters: [Int: [CheckedContinuation<Void, Never>]] = [:]
    private(set) var startedIDs: [Int] = []

    func recordStart(_ expertID: Int) {
        startedIDs.append(expertID)
        let waiters = startWaiters.removeValue(forKey: expertID) ?? []
        for waiter in waiters {
            waiter.resume()
        }
    }

    func waitUntilStarted(_ expertID: Int) async {
        guard !startedIDs.contains(expertID) else { return }
        await withCheckedContinuation { continuation in
            startWaiters[expertID, default: []].append(continuation)
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
        for waiter in waiters {
            waiter.resume()
        }
    }
}
