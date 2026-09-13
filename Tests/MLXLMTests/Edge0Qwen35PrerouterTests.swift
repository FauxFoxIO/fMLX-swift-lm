// Copyright © 2026 Faux Fox.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLLM

final class Edge0Qwen35PrerouterTests: XCTestCase {
    func testHeadUsesCurrentAndPreviousExecutedFeaturesWithExactGELU() throws {
        let configuration = fixtureConfiguration(owners: [0])
        let head = Edge0Qwen35PrerouterHead(configuration: configuration)
        try installFixtureWeights(on: head, configuration: configuration)

        let hidden = MLXArray([Float(0.25), -0.5]).reshaped(1, 1, 2)
        let current = MLXArray([Float(1), 0, 1, 0]).reshaped(1, 1, 4)
        let previous = MLXArray([Float(0), 1, 0, 1]).reshaped(1, 1, 4)

        let output = head(hidden: hidden, executed: current, previousExecuted: previous)
        let features = concatenated([hidden, current, previous], axis: -1).asType(.float16)
        let expected = head.linearInit(features) + head.fc2(MLXNN.gelu(head.fc1(features)))
        eval(output, expected)

        XCTAssertTrue(allClose(output, expected, rtol: 0, atol: 0).item(Bool.self))
        XCTAssertEqual(output.dtype, .float16)
    }

    func testStatePreservesPendingPredictionsAndTerminalConsumer() throws {
        let configuration = fixtureConfiguration(owners: [0, 2])
        let prerouter = Edge0Qwen35Prerouter(configuration: configuration)
        for head in prerouter.heads.values {
            try installFixtureWeights(on: head, configuration: configuration)
        }
        let state = Edge0Qwen35PrerouterState(configuration: configuration)
        let ownerZero = MLXArray([Int32(0), 2]).reshaped(1, 1, 2)
        let ownerTwo = MLXArray([Int32(1), 3]).reshaped(1, 1, 2)
        try state.recordExecuted(ownerZero, for: 0)
        try state.recordExecuted(ownerTwo, for: 2)

        try prerouter.stage(
            state: state,
            hiddenByOwner: [
                0: MLXArray([Float(0.25), -0.5]).reshaped(1, 1, 2),
                2: MLXArray([Float(-0.75), 0.5]).reshaped(1, 1, 2),
            ])

        XCTAssertNotNil(state.prediction(forConsumer: 1))
        XCTAssertNotNil(state.prediction(forConsumer: 3))
        XCTAssertNil(state.prediction(forConsumer: 0))

        let current = try state.currentExecuted(for: 0)
        state.swap()
        let previous = try state.previousExecuted(for: 0, matching: current)
        eval(current, previous)
        XCTAssertTrue(allClose(current, previous, rtol: 0, atol: 0).item(Bool.self))
        XCTAssertNotNil(state.prediction(forConsumer: 1))
        XCTAssertNotNil(state.prediction(forConsumer: 3))
    }

    func testLayerSixSeedsFourExpertPredictionsForTheNextTokenConsumers() throws {
        let configuration = Edge0Qwen35PrerouterConfiguration(
            layerCount: 40,
            hiddenSize: 2,
            expertCount: 8,
            topK: 4,
            prerouterHiddenSize: 3,
            owners: Array(6 ... 37))
        let prerouter = try Edge0Qwen35Prerouter.load(
            weights: fixtureWeights(configuration: configuration), configuration: configuration)
        let state = Edge0Qwen35PrerouterState(configuration: configuration)

        for layer in 6 ... 39 {
            try state.recordExecuted(executedIDs(offset: layer), for: layer)
        }
        try prerouter.stage(
            state: state,
            hiddenByOwner: Dictionary(
                uniqueKeysWithValues: (6 ... 37).map { layer in
                    (layer, MLXArray([Float(layer), -Float(layer)]).reshaped(1, 1, 2))
                }))

        for consumer in 7 ... 38 {
            let prediction = try XCTUnwrap(state.prediction(forConsumer: consumer))
            eval(prediction.expertIDs, prediction.scores)
            XCTAssertEqual(prediction.expertIDs.shape, [1, 1, 4])
            XCTAssertEqual(prediction.scores.shape, [1, 1, 4])
            XCTAssertEqual(Set(prediction.expertIDs.asArray(UInt32.self)).count, 4)
        }
        XCTAssertNil(state.prediction(forConsumer: 6))
        XCTAssertNil(state.prediction(forConsumer: 39))

        state.swap()
        XCTAssertNotNil(state.prediction(forConsumer: 7))
        XCTAssertNotNil(state.prediction(forConsumer: 38))
    }

    func testLayerSixPredictionUsesItsRealExecutedExperts() throws {
        let configuration = Edge0Qwen35PrerouterConfiguration(
            layerCount: 8,
            hiddenSize: 2,
            expertCount: 8,
            topK: 4,
            prerouterHiddenSize: 3,
            owners: [6])
        let prerouter = try Edge0Qwen35Prerouter.load(
            weights: fixtureWeights(configuration: configuration), configuration: configuration)

        func prediction(afterExecuting ids: [Int32]) throws -> MLXArray {
            let state = Edge0Qwen35PrerouterState(configuration: configuration)
            try state.recordExecuted(MLXArray(ids).reshaped(1, 1, 4), for: 6)
            try prerouter.stage(
                state: state,
                hiddenByOwner: [6: MLXArray([Float(0.5), -0.25]).reshaped(1, 1, 2)])
            return try XCTUnwrap(state.prediction(forConsumer: 7)).logits
        }

        let first = try prediction(afterExecuting: [0, 1, 2, 3])
        let second = try prediction(afterExecuting: [4, 5, 6, 7])
        eval(first, second)
        XCTAssertNotEqual(first.asArray(Float.self), second.asArray(Float.self))
    }

    func testArtifactLoaderUsesNamedSidecarAndConvertsFloatingWeightsToFP16() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configuration = fixtureConfiguration(owners: [0])
        let arrays = fixtureWeights(configuration: configuration)
        try save(
            arrays: arrays,
            metadata: [
                "__metadata__": """
                {"model":"edge0-35b","kind":"prerouter","K":"4","r":"16","alpha":"32","source":"fixture","source_md5":"df15dff55499f6dc0343d54379b03e32","converted":"2026-09-08T09:59:20+00:00","format_version":"1","owners":"[0]"}
                """
            ],
            url: directory.appendingPathComponent(Edge0Qwen35Prerouter.artifactFileName))

        let prerouter = try Edge0Qwen35Prerouter.load(
            from: directory, configuration: configuration)
        let head = try XCTUnwrap(prerouter.heads[0])
        XCTAssertEqual(head.fc1.weight.dtype, .float16)
        XCTAssertEqual(
            head.fc1.weight.shape, [configuration.prerouterHiddenSize, configuration.featureSize])
        XCTAssertEqual(
            head.fc2.weight.shape, [configuration.expertCount, configuration.prerouterHiddenSize])
        XCTAssertEqual(
            head.linearInit.weight.shape, [configuration.expertCount, configuration.featureSize])
    }

    func testArtifactLoaderRejectsInvalidWeightShape() {
        let configuration = fixtureConfiguration(owners: [0])
        var arrays = fixtureWeights(configuration: configuration)
        arrays["layers.0.fc1.weight"] = MLXArray.zeros([1, 1])

        XCTAssertThrowsError(
            try Edge0Qwen35Prerouter.load(weights: arrays, configuration: configuration)
        ) { error in
            XCTAssertEqual(
                error as? Edge0Qwen35PrerouterArtifactError,
                .invalidWeightShape(
                    name: "layers.0.fc1.weight",
                    expected: [configuration.prerouterHiddenSize, configuration.featureSize],
                    actual: [1, 1]))
        }
    }
}

private func fixtureConfiguration(owners: [Int]) -> Edge0Qwen35PrerouterConfiguration {
    Edge0Qwen35PrerouterConfiguration(
        layerCount: 4,
        hiddenSize: 2,
        expertCount: 4,
        topK: 2,
        prerouterHiddenSize: 3,
        owners: owners)
}

private func fixtureWeights(
    configuration: Edge0Qwen35PrerouterConfiguration
) -> [String: MLXArray] {
    Dictionary(
        uniqueKeysWithValues: configuration.owners.flatMap { owner in
            [
                (
                    "layers.\(owner).fc1.weight",
                    sequential(
                        configuration.prerouterHiddenSize * configuration.featureSize,
                        shape: [configuration.prerouterHiddenSize, configuration.featureSize])
                ),
                (
                    "layers.\(owner).fc2.weight",
                    sequential(
                        configuration.expertCount * configuration.prerouterHiddenSize,
                        shape: [configuration.expertCount, configuration.prerouterHiddenSize])
                ),
                (
                    "layers.\(owner).linear_init.weight",
                    sequential(
                        configuration.expertCount * configuration.featureSize,
                        shape: [configuration.expertCount, configuration.featureSize])
                ),
            ]
        })
}

private func installFixtureWeights(
    on head: Edge0Qwen35PrerouterHead,
    configuration: Edge0Qwen35PrerouterConfiguration
) throws {
    let weights = fixtureWeights(configuration: configuration)
    let owner = configuration.owners.first!
    try head.update(
        parameters: ModuleParameters.unflattened([
            "fc1.weight": weights["layers.\(owner).fc1.weight"]!.asType(.float16),
            "fc2.weight": weights["layers.\(owner).fc2.weight"]!.asType(.float16),
            "linear_init.weight": weights["layers.\(owner).linear_init.weight"]!.asType(.float16),
        ]),
        verify: [.all])
}

private func sequential(_ count: Int, shape: [Int]) -> MLXArray {
    MLXArray((0 ..< count).map { Float($0) * 0.01 - 0.1 }).reshaped(shape)
}

private func executedIDs(offset: Int) -> MLXArray {
    MLXArray((0 ..< 4).map { Int32(($0 + offset) % 8) }).reshaped(1, 1, 4)
}
