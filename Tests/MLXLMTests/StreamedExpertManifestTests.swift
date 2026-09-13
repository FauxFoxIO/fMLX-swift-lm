// Copyright © 2026 Faux Fox.

import Foundation
import MLXLMCommon
import XCTest

final class StreamedExpertManifestTests: XCTestCase {
    func testRoundTripsValidatedManifest() throws {
        let manifest = makeManifest()

        let data = try JSONEncoder().encode(manifest)
        let decoded = try JSONDecoder().decode(StreamedExpertManifest.self, from: data)

        XCTAssertEqual(decoded, manifest)
        XCTAssertEqual(decoded.formatVersion, StreamedExpertManifest.currentFormatVersion)
    }

    func testDecoderRejectsPathTraversal() throws {
        let data = try JSONEncoder().encode(
            makeManifest(source: .init(file: "../weights.bin", offset: 0, length: 8)))

        XCTAssertThrowsError(try JSONDecoder().decode(StreamedExpertManifest.self, from: data)) {
            XCTAssertEqual($0 as? StreamedExpertManifestError, .unsafeSourceFile("../weights.bin"))
        }
    }

    func testValidationRejectsOverflowedSourceRange() {
        let manifest = makeManifest(
            source: .init(file: "weights.bin", offset: .max, length: 1))

        XCTAssertThrowsError(try manifest.validate()) {
            XCTAssertEqual(
                $0 as? StreamedExpertManifestError, .sourceRangeOverflow(file: "weights.bin"))
        }
    }

    func testValidationRejectsOverflowedShape() {
        let manifest = makeManifest(shape: [.max, 2])

        XCTAssertThrowsError(try manifest.validate()) {
            XCTAssertEqual($0 as? StreamedExpertManifestError, .shapeElementCountOverflow)
        }
    }

    func testValidationRejectsDuplicateLayerProjectionExpertAndDenseEntries() {
        let slice = makeSlice()
        let duplicateLayer = StreamedExpertManifest(
            model: .init(id: "org/model", revision: "abc123"),
            layers: [
                .init(index: 0, projections: [.init(name: "gate_proj", experts: [slice])]),
                .init(index: 0, projections: []),
            ],
            denseResidency: [])
        assertValidationError(duplicateLayer, .duplicateLayerIndex(0))

        let duplicateProjection = StreamedExpertManifest(
            model: .init(id: "org/model", revision: "abc123"),
            layers: [
                .init(
                    index: 0,
                    projections: [
                        .init(name: "gate_proj", experts: [slice]),
                        .init(name: "gate_proj", experts: []),
                    ])
            ],
            denseResidency: [])
        assertValidationError(
            duplicateProjection, .duplicateProjection(layer: 0, name: "gate_proj"))

        let duplicateExpert = StreamedExpertManifest(
            model: .init(id: "org/model", revision: "abc123"),
            layers: [
                .init(
                    index: 0,
                    projections: [
                        .init(name: "gate_proj", experts: [slice, makeSlice()])
                    ])
            ],
            denseResidency: [])
        assertValidationError(
            duplicateExpert,
            .duplicateExpertSlice(
                layer: 0, projection: "gate_proj", index: 0, component: .weight))

        let dense = makeDenseTensor(name: "model.norm.weight")
        let duplicateDense = StreamedExpertManifest(
            model: .init(id: "org/model", revision: "abc123"),
            layers: [], denseResidency: [dense, dense])
        assertValidationError(duplicateDense, .duplicateDenseTensor("model.norm.weight"))
    }

    func testValidationRejectsOverlappingSourceRanges() {
        let first = makeSlice(source: .init(file: "weights.bin", offset: 0, length: 8))
        let second = makeSlice(index: 1, source: .init(file: "weights.bin", offset: 4, length: 8))
        let manifest = StreamedExpertManifest(
            model: .init(id: "org/model", revision: "abc123"),
            layers: [
                .init(index: 0, projections: [.init(name: "gate_proj", experts: [first, second])])
            ],
            denseResidency: [])

        assertValidationError(manifest, .overlappingSourceRanges(file: "weights.bin"))
    }

    func testValidationRejectsMalformedIntegrityAndQuantization() {
        let invalidIntegrity = makeManifest(integrity: .init(digest: "abcd"))
        assertValidationError(invalidIntegrity, .invalidSHA256Digest("abcd"))

        let invalidQuantization = makeManifest(
            quantization: .init(bits: 0, groupSize: 64, mode: "affine"))
        assertValidationError(invalidQuantization, .invalidQuantizationBits(0))
    }

    func testValidationRepresentsIndependentWeightScaleAndBiasRanges() throws {
        let components: [StreamedExpertManifest.ExpertSlice.Component] = [
            .weight, .scales, .biases,
        ]
        let projectionNames = ["gate_proj", "up_proj", "down_proj"]
        var offset: UInt64 = 0
        let projections = projectionNames.map { name in
            StreamedExpertManifest.Projection(
                name: name,
                experts: components.map { component in
                    defer { offset += 8 }
                    return .init(
                        index: 0,
                        component: component,
                        source: .init(file: "experts.safetensors", offset: offset, length: 8),
                        dtype: component == .weight ? "uint32" : "float16",
                        quantization: component == .weight
                            ? .init(bits: 4, groupSize: 32, mode: "affine") : nil,
                        shape: [1, 2, 2],
                        integrity: .init(digest: String(repeating: "c", count: 64)))
                })
        }
        let manifest = StreamedExpertManifest(
            model: .init(id: "org/model", revision: "abc123"),
            layers: [.init(index: 0, projections: projections)], denseResidency: [])

        XCTAssertNoThrow(try manifest.validate())
    }

    private func assertValidationError(
        _ manifest: StreamedExpertManifest,
        _ expected: StreamedExpertManifestError,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertThrowsError(try manifest.validate(), file: file, line: line) {
            XCTAssertEqual($0 as? StreamedExpertManifestError, expected, file: file, line: line)
        }
    }
}

private func makeManifest(
    source: StreamedExpertManifest.Source = .init(file: "experts.bin", offset: 0, length: 8),
    shape: [Int] = [2, 2],
    integrity: StreamedExpertManifest.Integrity = .init(digest: String(repeating: "a", count: 64)),
    quantization: StreamedExpertManifest.Quantization? = nil
) -> StreamedExpertManifest {
    StreamedExpertManifest(
        model: .init(id: "org/model", revision: "abc123"),
        layers: [
            .init(
                index: 0,
                projections: [
                    .init(
                        name: "gate_proj",
                        experts: [
                            makeSlice(
                                source: source, shape: shape, integrity: integrity,
                                quantization: quantization)
                        ])
                ])
        ],
        denseResidency: [makeDenseTensor()])
}

private func makeSlice(
    index: Int = 0,
    source: StreamedExpertManifest.Source = .init(file: "experts.bin", offset: 0, length: 8),
    shape: [Int] = [2, 2],
    integrity: StreamedExpertManifest.Integrity = .init(digest: String(repeating: "a", count: 64)),
    quantization: StreamedExpertManifest.Quantization? = nil
) -> StreamedExpertManifest.ExpertSlice {
    .init(
        index: index, source: source, dtype: "float16", quantization: quantization,
        shape: shape, integrity: integrity)
}

private func makeDenseTensor(
    name: String = "model.embed_tokens.weight"
) -> StreamedExpertManifest.DenseTensor {
    .init(
        name: name,
        source: .init(file: "dense.bin", offset: 0, length: 8),
        dtype: "float16",
        shape: [2, 2],
        integrity: .init(digest: String(repeating: "b", count: 64)))
}
