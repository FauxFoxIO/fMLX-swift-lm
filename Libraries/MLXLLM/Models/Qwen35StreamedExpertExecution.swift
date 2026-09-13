// Copyright © 2026 Faux Fox.

import MLXLMCommon

/// Errors raised before a streamed Qwen 3.5 MoE invocation reaches MLX.
///
/// A streamed call has a deliberately narrow contract: only decode and
/// two-token verification may compact the expert axis. Longer inputs keep the
/// resident expert layer because changing its width would change the prefill
/// kernels.
enum Qwen35StreamedExpertExecutionError: Error, Equatable, Sendable {
    case invalidExpertCount(Int)
    case invalidTopK(topK: Int, expertCount: Int)
    case requiresResidentExperts(sequenceLength: Int)
    case invalidRouterAssignmentCount(expected: Int, actual: Int)
    case invalidRouterExpertID(Int)
    case emptyCompactExpertPlan
    case mismatchedCompactExpertPayloadCount(expected: Int, actual: Int)
}

/// The compact expert axis and remapped router assignments for one decode call.
///
/// `compactRouterIDs` has the original row-major router order. Its values name
/// the first-occurrence slots in `expertIDs`, so repeated experts preserve the
/// exact assignment ordering while their weights load only once.
struct Qwen35CompactExpertPlan<Value>: @unchecked Sendable {
    let sequenceLength: Int
    let topK: Int
    let expertIDs: [Int]
    let compactRouterIDs: [UInt32]

    private let expertWeights: [ExpertWeight<Value>]

    var compactExpertCount: Int { expertIDs.count }

    /// Keeps non-Sendable MLX payloads inside the caller's execution domain.
    func withExpertWeights<Result>(
        _ body: ([ExpertWeight<Value>]) throws -> Result
    ) rethrows -> Result {
        try body(expertWeights)
    }

    init(
        sequenceLength: Int,
        topK: Int,
        expertIDs: [Int],
        compactRouterIDs: [UInt32],
        expertWeights: [ExpertWeight<Value>]
    ) throws {
        guard !expertIDs.isEmpty else {
            throw Qwen35StreamedExpertExecutionError.emptyCompactExpertPlan
        }
        guard expertIDs.count == expertWeights.count else {
            throw Qwen35StreamedExpertExecutionError.mismatchedCompactExpertPayloadCount(
                expected: expertIDs.count, actual: expertWeights.count)
        }
        self.sequenceLength = sequenceLength
        self.topK = topK
        self.expertIDs = expertIDs
        self.compactRouterIDs = compactRouterIDs
        self.expertWeights = expertWeights
    }
}

/// Serializes streamed-expert acquisition before Qwen 3.5 enters its MLX trace.
///
/// Router indices must be copied to host values outside the compiled MoE body.
/// This type performs only the asynchronous preparation phase; its returned
/// ``Qwen35CompactExpertPlan`` can then be consumed synchronously to build the
/// compact `[expert, ...]` tensors and run the existing SwitchGLU kernels.
final class Qwen35StreamedExpertExecution<Value>: @unchecked Sendable {
    private let store: ExpertWeightStore<Int, Value>
    private let preparation = SerialAccessContainer(())

    init(store: ExpertWeightStore<Int, Value>) {
        self.store = store
    }

    /// Acquires each unique routed expert in first-router-occurrence order.
    ///
    /// Inputs longer than two tokens are rejected before loading anything. The
    /// caller must run the unchanged resident layer for prefill instead.
    func prepareDecode(
        sequenceLength: Int,
        topK: Int,
        expertCount: Int,
        routerExpertIDs: [Int]
    ) async throws -> Qwen35CompactExpertPlan<Value> {
        try await preparation.read { [store] _ in
            try validate(
                sequenceLength: sequenceLength,
                topK: topK,
                expertCount: expertCount,
                routerExpertIDs: routerExpertIDs)

            var slotForExpert: [Int: UInt32] = [:]
            var expertIDs: [Int] = []
            var compactRouterIDs: [UInt32] = []
            expertIDs.reserveCapacity(routerExpertIDs.count)
            compactRouterIDs.reserveCapacity(routerExpertIDs.count)

            for expertID in routerExpertIDs {
                let slot: UInt32
                if let existing = slotForExpert[expertID] {
                    slot = existing
                } else {
                    slot = UInt32(expertIDs.count)
                    slotForExpert[expertID] = slot
                    expertIDs.append(expertID)
                }
                compactRouterIDs.append(slot)
            }

            var expertWeights: [ExpertWeight<Value>] = []
            expertWeights.reserveCapacity(expertIDs.count)
            for expertID in expertIDs {
                try Task.checkCancellation()
                expertWeights.append(try await store.expert(for: expertID))
            }

            return try Qwen35CompactExpertPlan(
                sequenceLength: sequenceLength,
                topK: topK,
                expertIDs: expertIDs,
                compactRouterIDs: compactRouterIDs,
                expertWeights: expertWeights)
        }
    }
}

private func validate(
    sequenceLength: Int,
    topK: Int,
    expertCount: Int,
    routerExpertIDs: [Int]
) throws {
    guard expertCount > 0 else {
        throw Qwen35StreamedExpertExecutionError.invalidExpertCount(expertCount)
    }
    guard topK > 0, topK <= expertCount else {
        throw Qwen35StreamedExpertExecutionError.invalidTopK(
            topK: topK, expertCount: expertCount)
    }
    guard (1 ... 2).contains(sequenceLength) else {
        throw Qwen35StreamedExpertExecutionError.requiresResidentExperts(
            sequenceLength: sequenceLength)
    }

    let (expectedCount, overflow) = sequenceLength.multipliedReportingOverflow(by: topK)
    guard !overflow, routerExpertIDs.count == expectedCount else {
        throw Qwen35StreamedExpertExecutionError.invalidRouterAssignmentCount(
            expected: overflow ? .max : expectedCount, actual: routerExpertIDs.count)
    }
    for expertID in routerExpertIDs where !(0 ..< expertCount).contains(expertID) {
        throw Qwen35StreamedExpertExecutionError.invalidRouterExpertID(expertID)
    }
}
