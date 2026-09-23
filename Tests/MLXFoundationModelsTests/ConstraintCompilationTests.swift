// Copyright © 2026 Faux Fox.

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

import Foundation
import MLXGuidedGeneration
import Synchronization
import Testing

@testable import MLXFoundationModels

extension FoundationModelsCacheTests {
    @Suite("Constraint compilation")
    struct ConstraintCompilation {
        @Test("a cold grammar does not block cached grammar hits")
        func coldCompilationAllowsCacheHits() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let tokenizer = try byteTokenizer()
            let modelID = "org/grammar-hot-\(UUID().uuidString)"
            let hotSource = #"{"type":"string"}"#
            let coldSource = #"{"type":"boolean"}"#
            _ = try await MLXLanguageModel.makeConstraint(
                modelID: modelID, kind: .json, source: hotSource
            ) {
                try GrammarConstraint(tokenizer: tokenizer, jsonSchema: hotSource)
            }

            let entered = DispatchSemaphore(value: 0)
            let release = DispatchSemaphore(value: 0)
            let hotFinished = Mutex(false)
            let cold = Task {
                try await MLXLanguageModel.makeConstraint(
                    modelID: modelID, kind: .json, source: coldSource
                ) {
                    entered.signal()
                    _ = release.wait(timeout: .now() + 3)
                    return try GrammarConstraint(tokenizer: tokenizer, jsonSchema: coldSource)
                }
            }
            let started = await Task.detached {
                waitForSignal(entered)
            }.value
            let hot = Task {
                _ = try await MLXLanguageModel.makeConstraint(
                    modelID: modelID, kind: .json, source: hotSource
                ) {
                    throw UnexpectedCompilation()
                }
                hotFinished.withLock { $0 = true }
            }

            try await Task.sleep(for: .milliseconds(200))
            let finishedBeforeRelease = hotFinished.withLock { $0 }
            release.signal()
            _ = try await cold.value
            try await hot.value
            #expect(started)
            #expect(finishedBeforeRelease)
        }

        @Test("concurrent callers share an uncached compile")
        func sharesCompilation() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let tokenizer = try byteTokenizer()
            let modelID = "org/grammar-shared-\(UUID().uuidString)"
            let source = #"{"type":"string"}"#
            let entered = DispatchSemaphore(value: 0)
            let release = DispatchSemaphore(value: 0)
            let count = Mutex(0)
            let compile: @Sendable () throws -> GrammarConstraint = {
                let invocation = count.withLock { value in
                    value += 1
                    return value
                }
                if invocation == 1 {
                    entered.signal()
                    _ = release.wait(timeout: .now() + 3)
                }
                return try GrammarConstraint(tokenizer: tokenizer, jsonSchema: source)
            }

            let first = Task {
                try await MLXLanguageModel.makeConstraint(
                    modelID: modelID, kind: .json, source: source, compile: compile)
            }
            let started = await Task.detached {
                waitForSignal(entered)
            }.value
            let second = Task {
                try await MLXLanguageModel.makeConstraint(
                    modelID: modelID, kind: .json, source: source, compile: compile)
            }
            try await Task.sleep(for: .milliseconds(100))
            release.signal()
            _ = try await first.value
            _ = try await second.value
            #expect(started)
            let observedCount = count.withLock { $0 }
            #expect(observedCount == 1)
        }

        @Test("a shared compilation failure does not poison later requests")
        func sharedFailureCanRetry() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let tokenizer = try byteTokenizer()
            let modelID = "org/grammar-failure-\(UUID().uuidString)"
            let source = #"{"type":"string"}"#
            let entered = DispatchSemaphore(value: 0)
            let release = DispatchSemaphore(value: 0)
            let count = Mutex(0)
            let compile: @Sendable () throws -> GrammarConstraint = {
                let invocation = count.withLock { value in
                    value += 1
                    return value
                }
                if invocation == 1 {
                    entered.signal()
                    _ = release.wait(timeout: .now() + 3)
                    throw DeliberateCompilationFailure()
                }
                return try GrammarConstraint(tokenizer: tokenizer, jsonSchema: source)
            }

            let first = Task {
                try await MLXLanguageModel.makeConstraint(
                    modelID: modelID, kind: .json, source: source, compile: compile)
            }
            let started = await Task.detached {
                waitForSignal(entered)
            }.value
            let second = Task {
                try await MLXLanguageModel.makeConstraint(
                    modelID: modelID, kind: .json, source: source, compile: compile)
            }
            try await Task.sleep(for: .milliseconds(100))
            release.signal()
            await #expect(throws: DeliberateCompilationFailure.self) { try await first.value }
            await #expect(throws: DeliberateCompilationFailure.self) { try await second.value }
            #expect(started)
            _ = try await MLXLanguageModel.makeConstraint(
                modelID: modelID, kind: .json, source: source, compile: compile)
            let observedCount = count.withLock { $0 }
            #expect(observedCount == 2)
        }

        @Test("eviction during compilation prevents stale cache insertion")
        func evictionDuringCompilation() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let tokenizer = try byteTokenizer()
            let modelID = "org/grammar-evict-\(UUID().uuidString)"
            let source = #"{"type":"string"}"#
            let entered = DispatchSemaphore(value: 0)
            let release = DispatchSemaphore(value: 0)
            let count = Mutex(0)
            let compile: @Sendable () throws -> GrammarConstraint = {
                let invocation = count.withLock { value in
                    value += 1
                    return value
                }
                if invocation == 1 {
                    entered.signal()
                    _ = release.wait(timeout: .now() + 3)
                }
                return try GrammarConstraint(tokenizer: tokenizer, jsonSchema: source)
            }

            let first = Task {
                try await MLXLanguageModel.makeConstraint(
                    modelID: modelID, kind: .json, source: source, compile: compile)
            }
            let started = await Task.detached {
                waitForSignal(entered)
            }.value
            await MLXLanguageModel.evictAll()
            release.signal()
            _ = try await first.value
            _ = try await MLXLanguageModel.makeConstraint(
                modelID: modelID, kind: .json, source: source, compile: compile)
            #expect(started)
            let observedCount = count.withLock { $0 }
            #expect(observedCount == 2)
        }

        @Test("cache keys separate model IDs and fast-forward modes")
        func distinctCacheKeys() async throws {
            guard #available(iOS 27.0, macOS 27.0, visionOS 27.0, *) else { return }
            let tokenizer = try byteTokenizer()
            let baseID = "org/grammar-key-\(UUID().uuidString)"
            let count = Mutex(0)
            let compile: @Sendable () throws -> GrammarConstraint = {
                count.withLock { $0 += 1 }
                return try GrammarConstraint(
                    tokenizer: tokenizer, jsonSchema: #"{"type":"string"}"#)
            }

            _ = try await MLXLanguageModel.makeConstraint(
                modelID: "\(baseID):json:x", kind: .json, source: "y", compile: compile)
            _ = try await MLXLanguageModel.makeConstraint(
                modelID: baseID, kind: .json, source: "x:json:y", compile: compile)
            _ = try await MLXLanguageModel.makeConstraint(
                modelID: baseID, kind: .json, source: "x:json:y",
                fastForward: true, compile: compile)

            let observedCount = count.withLock { $0 }
            #expect(observedCount == 3)
        }
    }
}

private struct UnexpectedCompilation: Error {}
private struct DeliberateCompilationFailure: Error {}

private func waitForSignal(_ semaphore: DispatchSemaphore) -> Bool {
    semaphore.wait(timeout: .now() + 2) == .success
}

private func byteTokenizer() throws -> GrammarTokenizer {
    let vocab = (0 ..< 256).map { String(format: "<0x%02X>", $0) }
    return try GrammarTokenizer(vocab: vocab, vocabType: .byteFallback, eosTokenId: 255)
}

#endif
