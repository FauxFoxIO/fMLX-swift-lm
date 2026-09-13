// Copyright © 2026 Faux Fox.

/// A platform-neutral memory-pressure signal supplied by an application.
///
/// An iOS application can map its platform notification to this value without
/// making an expert-streaming implementation depend on UIKit.
public enum MemoryPressureSignal: Sendable, Equatable {
    /// The process is not under memory pressure.
    case normal

    /// The process should reduce its resident expert budget.
    case warning

    /// The process should use its smallest resident expert budget immediately.
    case critical
}

/// The lifecycle state relevant to expert prefetching.
public enum ExpertStreamingActivityState: Sendable, Equatable {
    /// The application is allowed to prefetch according to its selected budget.
    case foreground

    /// The application must not start or retain speculative prefetches.
    case background
}

/// The resident and prefetched-expert limits for one pressure level.
public struct ExpertStreamingMemoryBudget: Sendable, Equatable {
    /// Maximum experts that may remain resident for streaming work.
    public let maximumResidentExpertCount: Int

    /// Maximum experts that may be prefetched ahead of demand.
    public let maximumPrefetchedExpertCount: Int

    /// Creates an expert-streaming memory budget.
    public init(
        maximumResidentExpertCount: Int,
        maximumPrefetchedExpertCount: Int
    ) throws {
        guard maximumResidentExpertCount >= 0 else {
            throw ExpertStreamingMemoryPressureError.negativeResidentExpertCount(
                maximumResidentExpertCount)
        }
        guard maximumPrefetchedExpertCount >= 0 else {
            throw ExpertStreamingMemoryPressureError.negativePrefetchedExpertCount(
                maximumPrefetchedExpertCount)
        }
        self.maximumResidentExpertCount = maximumResidentExpertCount
        self.maximumPrefetchedExpertCount = maximumPrefetchedExpertCount
    }
}

/// Configuration errors for ``ExpertStreamingMemoryPressureConfiguration``.
public enum ExpertStreamingMemoryPressureError: Error, Sendable, Equatable {
    /// A resident-expert count cannot be negative.
    case negativeResidentExpertCount(Int)

    /// A prefetched-expert count cannot be negative.
    case negativePrefetchedExpertCount(Int)

    /// Warning pressure cannot expand either expert budget.
    case warningBudgetExceedsNormal

    /// Critical pressure cannot expand either expert budget.
    case criticalBudgetExceedsWarning

    /// Critical pressure must stop speculative prefetch.
    case criticalBudgetAllowsPrefetch
}

/// Pressure-specific budgets for expert streaming.
///
/// Both warning and critical budgets must only shrink from the preceding level.
/// Critical budgets and background state suspend prefetch.
public struct ExpertStreamingMemoryPressureConfiguration: Sendable, Equatable {
    /// Budget used without memory pressure.
    public let normal: ExpertStreamingMemoryBudget

    /// Budget used after a warning memory-pressure signal.
    public let warning: ExpertStreamingMemoryBudget

    /// Budget used after a critical memory-pressure signal.
    public let critical: ExpertStreamingMemoryBudget

    /// Creates pressure-specific budgets that only become smaller as pressure rises.
    public init(
        normal: ExpertStreamingMemoryBudget,
        warning: ExpertStreamingMemoryBudget,
        critical: ExpertStreamingMemoryBudget
    ) throws {
        guard warning.isNoLargerThan(normal) else {
            throw ExpertStreamingMemoryPressureError.warningBudgetExceedsNormal
        }
        guard critical.isNoLargerThan(warning) else {
            throw ExpertStreamingMemoryPressureError.criticalBudgetExceedsWarning
        }
        guard critical.maximumPrefetchedExpertCount == 0 else {
            throw ExpertStreamingMemoryPressureError.criticalBudgetAllowsPrefetch
        }
        self.normal = normal
        self.warning = warning
        self.critical = critical
    }
}

extension ExpertStreamingMemoryBudget {
    fileprivate func isNoLargerThan(_ other: Self) -> Bool {
        maximumResidentExpertCount <= other.maximumResidentExpertCount
            && maximumPrefetchedExpertCount <= other.maximumPrefetchedExpertCount
    }
}

/// The current expert-streaming limits selected by pressure and lifecycle state.
public struct ExpertStreamingMemoryPressureDecision: Sendable, Equatable {
    /// Signal that selected the resident-expert budget.
    public let signal: MemoryPressureSignal

    /// Lifecycle state that can suspend prefetch.
    public let activityState: ExpertStreamingActivityState

    /// Maximum experts that may remain resident for streaming work.
    public let maximumResidentExpertCount: Int

    /// Maximum experts that may be prefetched ahead of demand.
    public let maximumPrefetchedExpertCount: Int

    /// Whether queued and in-flight speculative prefetch should be stopped.
    public let shouldSuspendPrefetch: Bool
}

/// A deterministic policy controller for an application's expert-streaming loop.
///
/// This type does not observe operating-system notifications itself. Feed it
/// application-owned pressure and lifecycle inputs, then apply the returned
/// decision to the streamer's residency and prefetch controls.
public struct ExpertStreamingMemoryPressureController: Sendable {
    /// The fixed budgets used to select the next decision.
    public let configuration: ExpertStreamingMemoryPressureConfiguration

    /// Most recent decision, starting at normal foreground operation.
    public private(set) var decision: ExpertStreamingMemoryPressureDecision

    /// Creates a controller with normal foreground operation as its initial state.
    public init(configuration: ExpertStreamingMemoryPressureConfiguration) {
        self.configuration = configuration
        decision = Self.makeDecision(
            configuration: configuration, signal: .normal, activityState: .foreground)
    }

    /// Updates the state and returns limits for the supplied pressure and lifecycle inputs.
    @discardableResult
    public mutating func update(
        signal: MemoryPressureSignal,
        activityState: ExpertStreamingActivityState
    ) -> ExpertStreamingMemoryPressureDecision {
        let next = Self.makeDecision(
            configuration: configuration, signal: signal, activityState: activityState)
        decision = next
        return next
    }

    private static func makeDecision(
        configuration: ExpertStreamingMemoryPressureConfiguration,
        signal: MemoryPressureSignal,
        activityState: ExpertStreamingActivityState
    ) -> ExpertStreamingMemoryPressureDecision {
        let budget: ExpertStreamingMemoryBudget
        switch signal {
        case .normal:
            budget = configuration.normal
        case .warning:
            budget = configuration.warning
        case .critical:
            budget = configuration.critical
        }

        let shouldSuspendPrefetch = signal == .critical || activityState == .background
        return ExpertStreamingMemoryPressureDecision(
            signal: signal,
            activityState: activityState,
            maximumResidentExpertCount: budget.maximumResidentExpertCount,
            maximumPrefetchedExpertCount: shouldSuspendPrefetch
                ? 0 : budget.maximumPrefetchedExpertCount,
            shouldSuspendPrefetch: shouldSuspendPrefetch)
    }
}
