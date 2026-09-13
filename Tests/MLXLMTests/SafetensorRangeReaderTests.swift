// Copyright © 2026 Apple Inc.

import Foundation
import XCTest

@testable import MLXLMCommon

final class SafetensorRangeReaderTests: XCTestCase {
    func testIndexesTensorsAndReadsLeadingDimensionSlice() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let payload = Data((0 ..< 28).map(UInt8.init))
        let url = directory.appendingPathComponent("weights.safetensors")
        try writeSafetensors(
            header: [
                "weights": [
                    "dtype": "F32",
                    "shape": [3, 2],
                    "data_offsets": [0, 24],
                ],
                "bias": [
                    "dtype": "F16",
                    "shape": [2],
                    "data_offsets": [24, 28],
                ],
                "__metadata__": ["format": "pt"],
            ],
            payload: payload,
            to: url)

        let reader = try SafetensorRangeReader(url: url)

        XCTAssertEqual(reader.tensors.map(\.name), ["weights", "bias"])
        let weights = try XCTUnwrap(reader.tensor(named: "weights"))
        XCTAssertEqual(weights.dataType, .float32)
        XCTAssertEqual(weights.shape, [3, 2])
        XCTAssertEqual(
            try reader.byteRange(for: "weights", leadingRange: 1 ..< 3),
            (weights.dataRange.lowerBound + 8) ..< (weights.dataRange.lowerBound + 24))
        let slice = try await reader.readSlice(named: "weights", leadingRange: 1 ..< 3)
        XCTAssertEqual(slice, Data(payload[8 ..< 24]))
    }

    func testRejectsInconsistentTensorByteCount() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appendingPathComponent("weights.safetensors")
        try writeSafetensors(
            header: [
                "weights": [
                    "dtype": "F32",
                    "shape": [2],
                    "data_offsets": [0, 7],
                ]
            ],
            payload: Data(repeating: 0, count: 7),
            to: url)

        XCTAssertThrowsError(try SafetensorRangeReader(url: url)) { error in
            XCTAssertEqual(
                error as? SafetensorRangeReaderError, .tensorByteCountMismatch("weights"))
        }
    }

    func testRejectsHeaderLengthThatCannotFitTheFile() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appendingPathComponent("weights.safetensors")
        var headerSize = UInt64.max.littleEndian
        try Data(bytes: &headerSize, count: MemoryLayout<UInt64>.size).write(to: url)

        XCTAssertThrowsError(try SafetensorRangeReader(url: url)) { error in
            XCTAssertEqual(error as? SafetensorRangeReaderError, .malformedHeader)
        }
    }

    func testRejectsShapeByteCountOverflow() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appendingPathComponent("weights.safetensors")
        try writeRawSafetensors(
            header: """
                {"weights":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,0]}}
                """,
            payload: Data(),
            to: url)

        XCTAssertThrowsError(try SafetensorRangeReader(url: url)) { error in
            XCTAssertEqual(error as? SafetensorRangeReaderError, .integerOverflow)
        }
    }

    func testRejectsLeadingRangeOutsideTensor() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appendingPathComponent("weights.safetensors")
        try writeSafetensors(
            header: [
                "weights": [
                    "dtype": "F32",
                    "shape": [2, 1],
                    "data_offsets": [0, 8],
                ]
            ],
            payload: Data(repeating: 0, count: 8),
            to: url)

        let reader = try SafetensorRangeReader(url: url)
        XCTAssertThrowsError(try reader.byteRange(for: "weights", leadingRange: 1 ..< 3)) { error in
            XCTAssertEqual(error as? SafetensorRangeReaderError, .invalidLeadingRange("weights"))
        }
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("SafetensorRangeReaderTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: false)
        return url
    }

    private func writeSafetensors(header: [String: Any], payload: Data, to url: URL) throws {
        let headerData = try JSONSerialization.data(withJSONObject: header)
        try writeSafetensors(headerData: headerData, payload: payload, to: url)
    }

    private func writeRawSafetensors(header: String, payload: Data, to url: URL) throws {
        try writeSafetensors(headerData: Data(header.utf8), payload: payload, to: url)
    }

    private func writeSafetensors(headerData: Data, payload: Data, to url: URL) throws {
        var headerSize = UInt64(headerData.count).littleEndian
        var data = Data(bytes: &headerSize, count: MemoryLayout<UInt64>.size)
        data.append(headerData)
        data.append(payload)
        try data.write(to: url)
    }
}
