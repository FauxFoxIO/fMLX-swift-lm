// Copyright © 2026 Faux Fox.

import Foundation
import MLXLMCommon

extension NativeTextModel {
    public struct TextProgress: Sendable, Hashable {
        public enum Phase: Sendable, Hashable {
            case prefill
            case generating
        }

        public let phase: Phase
        public let completedTokens: Int
        public let totalTokens: Int?
        public let generatedTokens: Int
        public let tokensPerSecond: Double?
        public let elapsedSeconds: Double
    }

    public enum TextEvent: Sendable {
        case text(String)
        case reasoning(String)
        case toolCall(ToolCall)
        case progress(TextProgress)
        case completed(
            inputTokens: Int, outputTokens: Int, cachedTokens: Int, reason: GenerateStopReason)
    }

    /// Each stream owns its decoder and protocol parsers; only immutable checkpoint data is shared.
    public func generateText(
        request: ConcurrentTextRuntime.Request, tools: [[String: any Sendable]] = []
    ) -> AsyncThrowingStream<TextEvent, Error> {
        AsyncThrowingStream(bufferingPolicy: .bufferingOldest(256)) { continuation in
            let task = Task {
                var generationID: UUID?
                do {
                    let generation = try await runtime.generate(request)
                    generationID = generation.id
                    let decoder = text.makeDecoder()
                    let processor = ToolCallProcessor(format: toolCallFormat ?? .json, tools: tools)
                    var reasoning = try reasoningConfig.map { config in
                        ReasoningEventEmitter(
                            config: config,
                            primedInside: ReasoningEventEmitter.promptEndsInsideReasoning(
                                renderedPromptTail: try text.decode(
                                    Array(request.tokens.suffix(128))),
                                config: config))
                    }
                    var outputTokens = 0
                    var cachedTokens = 0
                    let generationStartedAt = Date()
                    var decodeStartedAt: Date?
                    var lastProgressAt = Date.distantPast
                    func emit(_ event: TextEvent) throws {
                        try Task.checkCancellation()
                        switch continuation.yield(event) {
                        case .enqueued: break
                        case .dropped: throw ConcurrentTextRuntimeError.consumerTooSlow
                        case .terminated: throw CancellationError()
                        @unknown default: throw CancellationError()
                        }
                    }
                    func route(_ outputs: [ToolCallProcessor.Output]) throws {
                        for output in outputs {
                            switch output {
                            case .response(let value): try emit(.text(value))
                            case .toolCall(let call): try emit(.toolCall(call))
                            case .rejectedToolCall:
                                throw CheckpointTextError.invalidConfiguration(
                                    "Model emitted a malformed or unauthorized tool call")
                            }
                        }
                    }
                    func routeSegments(_ segments: [ReasoningEventEmitter.Segment]) throws {
                        for segment in segments {
                            switch segment {
                            case .response(let value):
                                try route(processor.processChunkOutputs(value))
                            case .reasoning(let value): try emit(.reasoning(value))
                            }
                        }
                    }
                    for try await event in generation.events {
                        try Task.checkCancellation()
                        switch event {
                        case .admitted(let count): cachedTokens = count
                        case .prefill(let processed, let total):
                            let elapsed = Date().timeIntervalSince(generationStartedAt)
                            try emit(
                                .progress(
                                    TextProgress(
                                        phase: .prefill,
                                        completedTokens: processed,
                                        totalTokens: total,
                                        generatedTokens: outputTokens,
                                        tokensPerSecond: nil,
                                        elapsedSeconds: elapsed
                                    )))
                        case .token(let token):
                            outputTokens += 1
                            let now = Date()
                            if decodeStartedAt == nil { decodeStartedAt = now }
                            let elapsed = max(
                                now.timeIntervalSince(decodeStartedAt ?? now),
                                0.001
                            )
                            if now.timeIntervalSince(lastProgressAt) >= 0.5 {
                                lastProgressAt = now
                                try emit(
                                    .progress(
                                        TextProgress(
                                            phase: .generating,
                                            completedTokens: outputTokens,
                                            totalTokens: request.maxTokens,
                                            generatedTokens: outputTokens,
                                            tokensPerSecond: Double(outputTokens) / elapsed,
                                            elapsedSeconds: elapsed
                                        )))
                            }
                            guard let chunk = try decoder.consume(token) else { continue }
                            if var scanner = reasoning {
                                try routeSegments(scanner.process(chunk))
                                reasoning = scanner
                            } else {
                                try route(processor.processChunkOutputs(chunk))
                            }
                        case .finished(let reason):
                            let elapsed = max(
                                Date().timeIntervalSince(
                                    decodeStartedAt ?? generationStartedAt
                                ),
                                0.001
                            )
                            try emit(
                                .progress(
                                    TextProgress(
                                        phase: .generating,
                                        completedTokens: outputTokens,
                                        totalTokens: request.maxTokens,
                                        generatedTokens: outputTokens,
                                        tokensPerSecond: Double(outputTokens) / elapsed,
                                        elapsedSeconds: elapsed
                                    )))
                            if var scanner = reasoning { try routeSegments(scanner.finalize()) }
                            try route(processor.processEOSOutputs())
                            try emit(
                                .completed(
                                    inputTokens: request.tokens.count, outputTokens: outputTokens,
                                    cachedTokens: cachedTokens, reason: reason))
                        default: break
                        }
                    }
                    continuation.finish()
                } catch {
                    if let generationID { await runtime.cancel(generationID) }
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
}
