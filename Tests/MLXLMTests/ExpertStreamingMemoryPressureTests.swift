// Copyright © 2026 Faux Fox.

import XCTest

@testable import MLXLMCommon

final class ExpertStreamingMemoryPressureTests: XCTestCase {
    func testBudgetRejectsNegativeCounts() {
        XCTAssertThrowsError(
            try ExpertStreamingMemoryBudget(
                maximumResidentExpertCount: -1, maximumPrefetchedExpertCount: 0)
        ) { error in
            XCTAssertEqual(
                error as? ExpertStreamingMemoryPressureError,
                .negativeResidentExpertCount(-1))
        }
        XCTAssertThrowsError(
            try ExpertStreamingMemoryBudget(
                maximumResidentExpertCount: 0, maximumPrefetchedExpertCount: -1)
        ) { error in
            XCTAssertEqual(
                error as? ExpertStreamingMemoryPressureError,
                .negativePrefetchedExpertCount(-1))
        }
    }

    func testConfigurationOnlyAllowsPressureToShrinkBudgets() throws {
        let normal = try budget(resident: 8, prefetched: 4)
        let warning = try budget(resident: 4, prefetched: 2)
        let critical = try budget(resident: 1, prefetched: 0)

        XCTAssertNoThrow(
            try ExpertStreamingMemoryPressureConfiguration(
                normal: normal, warning: warning, critical: critical))
        XCTAssertThrowsError(
            try ExpertStreamingMemoryPressureConfiguration(
                normal: normal,
                warning: try budget(resident: 9, prefetched: 2),
                critical: critical)
        ) { error in
            XCTAssertEqual(
                error as? ExpertStreamingMemoryPressureError,
                .warningBudgetExceedsNormal)
        }
        XCTAssertThrowsError(
            try ExpertStreamingMemoryPressureConfiguration(
                normal: normal,
                warning: warning,
                critical: try budget(resident: 5, prefetched: 0))
        ) { error in
            XCTAssertEqual(
                error as? ExpertStreamingMemoryPressureError,
                .criticalBudgetExceedsWarning)
        }
        XCTAssertThrowsError(
            try ExpertStreamingMemoryPressureConfiguration(
                normal: normal,
                warning: warning,
                critical: try budget(resident: 1, prefetched: 1))
        ) { error in
            XCTAssertEqual(
                error as? ExpertStreamingMemoryPressureError,
                .criticalBudgetAllowsPrefetch)
        }
    }

    func testPressureTransitionsSelectShrunkenBudgets() throws {
        var controller = ExpertStreamingMemoryPressureController(configuration: try configuration())

        XCTAssertEqual(
            controller.decision,
            decision(
                signal: .normal, activity: .foreground, resident: 8, prefetched: 4,
                suspended: false))

        XCTAssertEqual(
            controller.update(signal: .warning, activityState: .foreground),
            decision(
                signal: .warning, activity: .foreground, resident: 4, prefetched: 2,
                suspended: false))
        XCTAssertEqual(
            controller.update(signal: .critical, activityState: .foreground),
            decision(
                signal: .critical, activity: .foreground, resident: 1, prefetched: 0,
                suspended: true))
        XCTAssertEqual(
            controller.update(signal: .normal, activityState: .foreground),
            decision(
                signal: .normal, activity: .foreground, resident: 8, prefetched: 4,
                suspended: false))
    }

    func testBackgroundSuspendsPrefetchWithoutChangingSelectedPressureBudget() throws {
        var controller = ExpertStreamingMemoryPressureController(configuration: try configuration())

        XCTAssertEqual(
            controller.update(signal: .normal, activityState: .background),
            decision(
                signal: .normal, activity: .background, resident: 8, prefetched: 0,
                suspended: true))
        XCTAssertEqual(
            controller.update(signal: .warning, activityState: .background),
            decision(
                signal: .warning, activity: .background, resident: 4, prefetched: 0,
                suspended: true))
        XCTAssertEqual(
            controller.update(signal: .warning, activityState: .foreground),
            decision(
                signal: .warning, activity: .foreground, resident: 4, prefetched: 2,
                suspended: false))
    }

    private func configuration() throws -> ExpertStreamingMemoryPressureConfiguration {
        try ExpertStreamingMemoryPressureConfiguration(
            normal: budget(resident: 8, prefetched: 4),
            warning: budget(resident: 4, prefetched: 2),
            critical: budget(resident: 1, prefetched: 0))
    }

    private func budget(resident: Int, prefetched: Int) throws -> ExpertStreamingMemoryBudget {
        try ExpertStreamingMemoryBudget(
            maximumResidentExpertCount: resident,
            maximumPrefetchedExpertCount: prefetched)
    }

    private func decision(
        signal: MemoryPressureSignal,
        activity: ExpertStreamingActivityState,
        resident: Int,
        prefetched: Int,
        suspended: Bool
    ) -> ExpertStreamingMemoryPressureDecision {
        .init(
            signal: signal,
            activityState: activity,
            maximumResidentExpertCount: resident,
            maximumPrefetchedExpertCount: prefetched,
            shouldSuspendPrefetch: suspended)
    }
}
