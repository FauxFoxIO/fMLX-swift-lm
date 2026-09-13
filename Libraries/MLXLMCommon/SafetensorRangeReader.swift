// Copyright © 2026 Apple Inc.

import Foundation

/// A data type encoded by a safetensors tensor.
public enum SafetensorDataType: String, CaseIterable, Sendable {
    case bool = "BOOL"
    case float4 = "F4"
    case float6E2M3 = "F6_E2M3"
    case float6E3M2 = "F6_E3M2"
    case uint8 = "U8"
    case int8 = "I8"
    case uint16 = "U16"
    case int16 = "I16"
    case uint32 = "U32"
    case int32 = "I32"
    case uint64 = "U64"
    case int64 = "I64"
    case float16 = "F16"
    case bfloat16 = "BF16"
    case float32 = "F32"
    case float64 = "F64"
    case complex64 = "C64"
    case complex128 = "C128"
    case float8E4M3 = "F8_E4M3"
    case float8E8M0 = "F8_E8M0"
    case float8E4M3FN = "F8_E4M3FN"
    case float8E4M3FNUZ = "F8_E4M3FNUZ"
    case float8E5M2 = "F8_E5M2"
    case float8E5M2FNUZ = "F8_E5M2FNUZ"
    case float8E8M0FNU = "F8_E8M0FNU"

    public var bitWidth: UInt64 {
        switch self {
        case .float4:
            4
        case .float6E2M3, .float6E3M2:
            6
        case .bool, .uint8, .int8, .float8E4M3, .float8E4M3FN, .float8E4M3FNUZ,
            .float8E5M2, .float8E5M2FNUZ, .float8E8M0, .float8E8M0FNU:
            8
        case .uint16, .int16, .float16, .bfloat16:
            16
        case .uint32, .int32, .float32:
            32
        case .uint64, .int64, .float64, .complex64:
            64
        case .complex128:
            128
        }
    }

    /// The bytes per element when the type is not bit-packed.
    public var byteWidth: UInt64? {
        bitWidth.isMultiple(of: 8) ? bitWidth / 8 : nil
    }
}

/// A tensor's validated safetensors metadata and absolute file range.
public struct SafetensorTensor: Sendable, Equatable {
    public let name: String
    public let dataType: SafetensorDataType
    public let shape: [UInt64]
    public let dataRange: Range<UInt64>

    public var byteCount: UInt64 {
        dataRange.upperBound - dataRange.lowerBound
    }
}

/// Errors raised while indexing or reading a safetensors file.
public enum SafetensorRangeReaderError: Error, Sendable, Equatable {
    case malformedHeader
    case unsupportedDataType(String)
    case invalidTensor(String)
    case tensorByteCountMismatch(String)
    case invalidDataLayout
    case integerOverflow
    case tensorNotFound(String)
    case tensorHasNoLeadingDimension(String)
    case invalidLeadingRange(String)
    case requestedRangeTooLarge
    case truncatedRead
}

/// Indexes a safetensors header and reads contiguous slices on demand.
///
/// Each read opens a separate file handle, so callers can request different tensor ranges
/// concurrently without synchronizing file offsets.
public struct SafetensorRangeReader: Sendable {
    private static let prefixByteCount: UInt64 = 8
    private let url: URL
    private let tensorsByName: [String: SafetensorTensor]

    public let tensors: [SafetensorTensor]

    /// Reads and validates only the safetensors prefix and JSON header.
    public init(url: URL, maximumHeaderSize: UInt64 = 512 * 1_024 * 1_024)
        throws
    {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }

        let fileSize = try handle.seekToEnd()
        guard fileSize >= Self.prefixByteCount else {
            throw SafetensorRangeReaderError.malformedHeader
        }
        try handle.seek(toOffset: 0)

        guard let prefix = try handle.read(upToCount: Int(Self.prefixByteCount)),
            prefix.count == Int(Self.prefixByteCount)
        else {
            throw SafetensorRangeReaderError.malformedHeader
        }
        let headerSize = prefix.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        let availableHeaderBytes = fileSize - Self.prefixByteCount
        guard headerSize <= maximumHeaderSize,
            headerSize <= availableHeaderBytes,
            headerSize <= UInt64(Int.max)
        else {
            throw SafetensorRangeReaderError.malformedHeader
        }

        guard let headerData = try handle.read(upToCount: Int(headerSize)),
            headerData.count == Int(headerSize)
        else {
            throw SafetensorRangeReaderError.malformedHeader
        }

        let header: [String: HeaderEntry]
        do {
            header = try JSONDecoder().decode([String: HeaderEntry].self, from: headerData)
        } catch {
            throw SafetensorRangeReaderError.malformedHeader
        }

        let dataStart = try Self.adding(Self.prefixByteCount, headerSize)
        let dataByteCount = fileSize - dataStart
        var indexed = [(tensor: SafetensorTensor, relativeRange: Range<UInt64>)]()
        indexed.reserveCapacity(header.count)

        for (name, entry) in header where name != "__metadata__" {
            guard let typeName = entry.dataType,
                let dataType = SafetensorDataType(rawValue: typeName)
            else {
                if let typeName = entry.dataType {
                    throw SafetensorRangeReaderError.unsupportedDataType(typeName)
                }
                throw SafetensorRangeReaderError.invalidTensor(name)
            }
            guard let shape = entry.shape,
                let offsets = entry.dataOffsets,
                offsets.count == 2,
                offsets[0] <= offsets[1],
                offsets[1] <= dataByteCount
            else {
                throw SafetensorRangeReaderError.invalidTensor(name)
            }

            let expectedByteCount = try Self.byteCount(shape: shape, dataType: dataType)
            guard offsets[1] - offsets[0] == expectedByteCount else {
                throw SafetensorRangeReaderError.tensorByteCountMismatch(name)
            }
            let lowerBound = try Self.adding(dataStart, offsets[0])
            let upperBound = try Self.adding(dataStart, offsets[1])
            indexed.append(
                (
                    SafetensorTensor(
                        name: name,
                        dataType: dataType,
                        shape: shape,
                        dataRange: lowerBound ..< upperBound),
                    offsets[0] ..< offsets[1]
                ))
        }

        indexed.sort {
            if $0.relativeRange.lowerBound == $1.relativeRange.lowerBound {
                return $0.tensor.name < $1.tensor.name
            }
            return $0.relativeRange.lowerBound < $1.relativeRange.lowerBound
        }

        var expectedOffset: UInt64 = 0
        for item in indexed {
            guard item.relativeRange.lowerBound == expectedOffset else {
                throw SafetensorRangeReaderError.invalidDataLayout
            }
            expectedOffset = item.relativeRange.upperBound
        }
        guard expectedOffset == dataByteCount else {
            throw SafetensorRangeReaderError.invalidDataLayout
        }

        let tensors = indexed.map(\.tensor)
        self.url = url
        self.tensors = tensors
        self.tensorsByName = Dictionary(uniqueKeysWithValues: tensors.map { ($0.name, $0) })
    }

    public func tensor(named name: String) -> SafetensorTensor? {
        tensorsByName[name]
    }

    /// Returns the file range for a contiguous selection of the first tensor dimension.
    public func byteRange(for tensorName: String, leadingRange: Range<UInt64>) throws -> Range<
        UInt64
    > {
        guard let tensor = tensorsByName[tensorName] else {
            throw SafetensorRangeReaderError.tensorNotFound(tensorName)
        }
        guard let leadingDimension = tensor.shape.first else {
            throw SafetensorRangeReaderError.tensorHasNoLeadingDimension(tensorName)
        }
        guard leadingRange.upperBound <= leadingDimension else {
            throw SafetensorRangeReaderError.invalidLeadingRange(tensorName)
        }
        guard !tensor.shape.contains(0) else {
            return tensor.dataRange.lowerBound ..< tensor.dataRange.lowerBound
        }

        let rowBitCount = try Self.bitCount(
            shape: Array(tensor.shape.dropFirst()), dataType: tensor.dataType)
        let startBit = try Self.multiplied(leadingRange.lowerBound, rowBitCount)
        let bitCount = try Self.multiplied(
            leadingRange.upperBound - leadingRange.lowerBound, rowBitCount)
        guard startBit.isMultiple(of: 8), bitCount.isMultiple(of: 8) else {
            throw SafetensorRangeReaderError.invalidLeadingRange(tensorName)
        }
        let startOffset = startBit / 8
        let byteCount = bitCount / 8
        let lowerBound = try Self.adding(tensor.dataRange.lowerBound, startOffset)
        let upperBound = try Self.adding(lowerBound, byteCount)
        guard upperBound <= tensor.dataRange.upperBound else {
            throw SafetensorRangeReaderError.invalidLeadingRange(tensorName)
        }
        return lowerBound ..< upperBound
    }

    /// Reads a first-dimension slice without materializing the rest of the tensor.
    ///
    /// Cancellation is checked before opening, after seeking, and after reading. The underlying
    /// synchronous file read cannot be interrupted by Foundation once it has begun.
    public func readSlice(named tensorName: String, leadingRange: Range<UInt64>) async throws
        -> Data
    {
        let range = try byteRange(for: tensorName, leadingRange: leadingRange)
        let byteCount = range.upperBound - range.lowerBound
        let url = self.url
        guard byteCount <= UInt64(Int.max) else {
            throw SafetensorRangeReaderError.requestedRangeTooLarge
        }

        try Task.checkCancellation()
        let readTask = Task.detached(priority: .utility) { () throws -> Data in
            try Task.checkCancellation()
            let handle = try FileHandle(forReadingFrom: url)
            defer { try? handle.close() }
            try handle.seek(toOffset: range.lowerBound)
            try Task.checkCancellation()
            guard let data = try handle.read(upToCount: Int(byteCount)),
                data.count == Int(byteCount)
            else {
                throw SafetensorRangeReaderError.truncatedRead
            }
            try Task.checkCancellation()
            return data
        }
        return try await withTaskCancellationHandler(
            operation: { try await readTask.value },
            onCancel: { readTask.cancel() })
    }

    private struct HeaderEntry: Decodable {
        let dataType: String?
        let shape: [UInt64]?
        let dataOffsets: [UInt64]?

        enum CodingKeys: String, CodingKey {
            case dataType = "dtype"
            case shape
            case dataOffsets = "data_offsets"
        }
    }

    private static func byteCount(shape: [UInt64], dataType: SafetensorDataType) throws -> UInt64 {
        let bitCount = try bitCount(shape: shape, dataType: dataType)
        guard bitCount.isMultiple(of: 8) else {
            throw SafetensorRangeReaderError.invalidDataLayout
        }
        return bitCount / 8
    }

    private static func bitCount(shape: [UInt64], dataType: SafetensorDataType) throws -> UInt64 {
        if shape.contains(0) { return 0 }
        let elementCount = try shape.reduce(UInt64(1)) { partialResult, dimension in
            try multiplied(partialResult, dimension)
        }
        return try multiplied(elementCount, dataType.bitWidth)
    }

    private static func adding(_ lhs: UInt64, _ rhs: UInt64) throws -> UInt64 {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        guard !overflow else { throw SafetensorRangeReaderError.integerOverflow }
        return value
    }

    private static func multiplied(_ lhs: UInt64, _ rhs: UInt64) throws -> UInt64 {
        let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
        guard !overflow else { throw SafetensorRangeReaderError.integerOverflow }
        return value
    }
}
