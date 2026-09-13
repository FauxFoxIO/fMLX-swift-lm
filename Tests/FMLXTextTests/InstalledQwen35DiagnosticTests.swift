// Copyright © 2026 Faux Fox.

import FMLXText
import Foundation
import MLXLLM
import MLXLMCommon
import Testing

@Suite("Installed Qwen 3.5-family diagnostics", .serialized)
struct InstalledQwen35DiagnosticTests {
    private static let installedModelPath: String? = {
        if let override = ProcessInfo.processInfo.environment["FMLX_INSTALLED_QWEN_PATH"] {
            return override
        }
        guard
            let support = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first
        else { return nil }
        return support.appending(
            path: "Flow/Models/OsaurusAI/Qwen3.6-35B-A3B-MXFP4-MTP",
            directoryHint: .isDirectory
        ).path
    }()

    @Test(
        "Exact installed checkpoint emits coherent text",
        .enabled(if: installedModelPath.map { FileManager.default.fileExists(atPath: $0) } == true)
    )
    func exactInstalledCheckpoint() async throws {
        let environment = ProcessInfo.processInfo.environment
        let path = try #require(Self.installedModelPath)
        let directory = URL(filePath: path, directoryHint: .isDirectory)
        let mode = environment["FMLX_INSTALLED_QWEN_MODE"] ?? "mtp"
        let usesMTP = mode == "mtp"

        let text = try await CheckpointTextProcessor.load(directory: directory)
        let target = try await NativeTextModelLoader.load(directory: directory)
        let drafter =
            usesMTP
            ? try await NativeTextModelLoader.loadCombinedMTP(directory: directory) : nil
        let physicalMemory = Int(ProcessInfo.processInfo.physicalMemory)
        let runtime = try ConcurrentTextRuntime(
            model: target,
            identity: try text.cacheIdentity(
                modelRevision: "installed-diagnostic",
                cacheLayoutRevision: "installed-diagnostic-native"
            ),
            configuration: .init(
                memoryBudgetBytes: physicalMemory - 4 * 1_024 * 1_024 * 1_024,
                prefixCacheBytes: 256 * 1_024 * 1_024,
                workingMemoryBytes: 1_024 * 1_024 * 1_024,
                maxActiveRequests: 1,
                maxQueuedRequests: 1,
                maxPromptTokens: 4_096,
                maxOutputTokens: 128,
                prefillChunkSize: 128,
                streamBufferSize: 256,
                batchDecode: true
            ),
            drafter: drafter
        )
        let prompt = try text.prepareChat(
            messages: [
                [
                    "role": "system",
                    "content":
                        "You are a precise Swift coding assistant. Reason and answer in English.",
                ],
                [
                    "role": "user",
                    "content": "Explain in one paragraph why a Swift actor prevents data races.",
                ],
            ],
            additionalContext: ["enable_thinking": true]
        )
        let request = ConcurrentTextRuntime.Request(
            tokens: prompt,
            maxTokens: 96,
            temperature: 0.6,
            topP: 0.95,
            topK: 20,
            seed: 7,
            stopTokenIDs: text.stopTokenIDs,
            speculative: usesMTP
        )

        let startedAt = ContinuousClock.now
        let generation = try await runtime.generate(request)
        var generatedTokens: [Int] = []
        var firstTokenAt: ContinuousClock.Instant?
        var finishedAt = startedAt
        var telemetry: SpeculativeDecodingTelemetry?
        var executionMode: ConcurrentTextRuntime.ExecutionMode?
        for try await event in generation.events {
            switch event {
            case .token(let token):
                if firstTokenAt == nil { firstTokenAt = .now }
                generatedTokens.append(token)
            case .execution(let mode):
                executionMode = mode
            case .speculation(let value):
                telemetry = value
            case .finished:
                finishedAt = .now
            default:
                break
            }
        }
        await runtime.shutdown()

        let output = try text.decode(generatedTokens)
        let timeToFirstToken = firstTokenAt.map { startedAt.duration(to: $0) }
        let decodeDuration = firstTokenAt.map { $0.duration(to: finishedAt) }
        let decodeSeconds = decodeDuration.map(Self.seconds) ?? 0
        let tokensPerSecond =
            decodeSeconds > 0
            ? Double(max(generatedTokens.count - 1, 0)) / decodeSeconds : 0
        print(
            "[InstalledQwen35Diagnostic] mode=\(mode) execution=\(String(describing: executionMode)) tokens=\(generatedTokens.count) ttft=\(String(describing: timeToFirstToken)) decode_tps=\(String(format: "%.2f", tokensPerSecond)) proposed=\(telemetry?.draftTokenCount ?? 0) accepted=\(telemetry?.acceptedDraftTokenCount ?? 0)"
        )
        print("[InstalledQwen35Diagnostic] output=\(output)")

        #expect(!generatedTokens.isEmpty)
        #expect(!output.localizedCaseInsensitiveContains("Kooper"))
        if usesMTP {
            #expect((telemetry?.draftTokenCount ?? 0) > 0)
            #expect((telemetry?.acceptedDraftTokenCount ?? 0) > 0)
        }
    }

    private static func seconds(_ duration: Duration) -> Double {
        let parts = duration.components
        return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
    }
}
