// Copyright © 2026 Faux Fox.

import MLX
import XCTest

@testable import MLXLLM

final class Edge0Qwen35InferenceProfileTests: XCTestCase {
    func testPinnedProfileKeepsTerminalLayerOnRealRouter() throws {
        let profile = Edge0Qwen35InferenceProfile.edge0_35b
        try profile.validate()

        XCTAssertEqual(profile.identity.modelID, "Edge0/Edge0-35B-A3B-preview")
        XCTAssertEqual(profile.identity.modelRevision, "1ff9f4478890faec0368c5463b621d1036d5b518")
        XCTAssertEqual(
            profile.identity.implementationRevision, "ae1ee2d343f88d6a9d3c9304d6a853353041a382")
        XCTAssertEqual(profile.artifactOwners, Array(6 ... 38))
        XCTAssertEqual(profile.predictedConsumers, Array(7 ... 38))
        XCTAssertEqual(profile.terminalRouterLayer, 39)
        XCTAssertFalse(profile.predictedConsumers.contains(profile.terminalRouterLayer))
        XCTAssertEqual(profile.topK, 4)
        XCTAssertFalse(profile.supportsMTP)
        XCTAssertFalse(profile.supportsBatching)
    }

    func testProfileRejectsMTPAndBatching() {
        let profile = Edge0Qwen35InferenceProfile.edge0_35b
        XCTAssertThrowsError(try profile.validateRequest(batchSize: 2, mtpEnabled: false)) {
            XCTAssertEqual($0 as? Edge0Qwen35InferenceProfileError, .batchUnsupported(2))
        }
        XCTAssertThrowsError(try profile.validateRequest(batchSize: 1, mtpEnabled: true)) {
            XCTAssertEqual($0 as? Edge0Qwen35InferenceProfileError, .mtpUnsupported)
        }
    }

    func testRequestStateRollsExecutedAndPredictedStateThenResets() throws {
        let state = try Edge0Qwen35InferenceRequestState()
        let executed = MLXArray([Int32(1), 7, 11, 31]).reshaped(1, 1, 4)
        try state.recordExecuted(executed, for: 6)
        state.advance()

        let previous = try XCTUnwrap(state.previousExecutedOneHot(for: 6))
        eval(previous)
        XCTAssertEqual(previous.shape, [1, 1, 256])
        XCTAssertEqual(previous.reshaped(-1).asArray(Float.self).filter { $0 == 1 }.count, 4)
        XCTAssertThrowsError(try state.previousPredictedOneHot(forConsumer: 39)) {
            XCTAssertEqual($0 as? Edge0Qwen35InferenceProfileError, .invalidPredictedConsumer(39))
        }

        state.resetForPrefixReuse()
        XCTAssertEqual(state.disposition, .prefixReset)
        XCTAssertNil(state.previousExecutedOneHot(for: 6))
        state.resetForCheckpointRestore()
        XCTAssertEqual(state.disposition, .checkpointReset)
    }

    func testRequestAndPrefillBoundariesClearTransientPrerouterState() throws {
        let state = try Edge0Qwen35InferenceRequestState()
        let executed = MLXArray([Int32(1), 7, 11, 31]).reshaped(1, 1, 4)
        try state.recordExecuted(executed, for: 6)
        state.advance()
        XCTAssertNotNil(state.previousExecutedOneHot(for: 6))

        // A prefetched prompt cache carries no prerouter tensors into the
        // first decode token, so it must begin with an empty feature history.
        state.resetForPrefixReuse()
        XCTAssertNil(state.previousExecutedOneHot(for: 6))
        XCTAssertNil(try state.previousPredictedOneHot(forConsumer: 7))

        // The same reset is required when an interrupted request releases
        // its model slot before another request begins.
        try state.recordExecuted(executed, for: 6)
        state.advance()
        state.resetForNewRequest()
        XCTAssertEqual(state.disposition, .fresh)
        XCTAssertNil(state.previousExecutedOneHot(for: 6))
        XCTAssertNil(try state.previousPredictedOneHot(forConsumer: 7))
    }

    func testPinnedPrefillPhaseKeepsMultiTokenChunksOnRealRouting() {
        func prefillPhases(promptTokenCount: Int) -> [Edge0Qwen35ForwardPhase] {
            let controller = Edge0Qwen35PhaseController()
            controller.begin(promptTokenCount: promptTokenCount)
            return (0 ..< promptTokenCount).map { _ in
                controller.prefillPhase(forwardTokenCount: 1)
            }
        }

        XCTAssertEqual(
            prefillPhases(promptTokenCount: 2), [.prefillRealRouter, .prefillRealRouter])
        XCTAssertEqual(prefillPhases(promptTokenCount: 1), [.prefillDecodeLike])

        let fullChunk = prefillPhases(promptTokenCount: 2048)
        XCTAssertEqual(fullChunk.count, 2048)
        XCTAssertTrue(fullChunk.allSatisfy { $0 == .prefillRealRouter })

        let onePastChunk = prefillPhases(promptTokenCount: 2049)
        XCTAssertEqual(
            Array(onePastChunk.prefix(2048)),
            Array(repeating: .prefillRealRouter, count: 2048))
        XCTAssertEqual(onePastChunk.last, .prefillDecodeLike)
    }

    func testStabilizerClampsAndScrubsNaNs() {
        let hidden = MLXArray([
            Float.nan, Float.infinity, -Float.infinity, -1_200, -999, 999, 1_200,
        ])
        let stabilized = Edge0Qwen35InferenceProfile.edge0_35b.stabilizedHidden(hidden)
        eval(stabilized)
        XCTAssertEqual(stabilized.asArray(Float.self), [0, 1000, -1000, -1000, -999, 999, 1000])
    }

    func testAttachmentRejectsAnotherPrerouterLayout() throws {
        let configuration = Edge0Qwen35PrerouterConfiguration(
            layerCount: 4, hiddenSize: 2, expertCount: 4, topK: 2,
            prerouterHiddenSize: 3, owners: [0])
        let prerouter = Edge0Qwen35Prerouter(configuration: configuration)
        XCTAssertThrowsError(
            try Edge0Qwen35InferenceProfile.edge0_35b.validateAttachment(prerouter)
        ) {
            XCTAssertEqual(
                $0 as? Edge0Qwen35InferenceProfileError,
                .prerouterConfigurationMismatch)
        }
    }

    func testProfileOnlyLoadsTheNamedSidecar() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertThrowsError(
            try Edge0Qwen35InferenceProfile.edge0_35b.loadPrerouter(from: directory)
        ) {
            XCTAssertEqual(
                $0 as? Edge0Qwen35PrerouterArtifactError,
                .missingArtifact(
                    directory.appendingPathComponent(Edge0Qwen35Prerouter.artifactFileName)))
        }
    }

    func testPrerouterMetadataRequiresPinnedKindAndOwners() throws {
        let metadata = [
            "__metadata__": """
            {"model":"edge0-35b","kind":"prerouter","K":"4","r":"16","alpha":"32","source":"fixture","source_md5":"df15dff55499f6dc0343d54379b03e32","converted":"2026-09-08T09:59:20+00:00","format_version":"1","owners":"[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]"}
            """
        ]
        XCTAssertNoThrow(
            try validateEdge0Qwen35ArtifactMetadata(
                metadata, kind: .prerouter, owners: Array(6 ... 38)))
        XCTAssertThrowsError(
            try validateEdge0Qwen35ArtifactMetadata(
                metadata, kind: .prerouter, owners: Array(6 ... 37))
        ) {
            XCTAssertEqual(
                $0 as? Edge0Qwen35ArtifactMetadataError,
                .incompatibleMetadata)
        }
    }
}
