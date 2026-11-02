// Copyright © 2026 Faux Fox, Inc.

import Darwin
import Foundation
import MLX

// Memory-mapped safetensors, the way llama.cpp loads weights on Apple silicon: the file is
// mapped read-only, page-aligned windows of the mapping become no-copy Metal buffers, and each
// tensor is a view at its byte offset. The weights stay clean, file-backed pages: they are not
// part of the process's `phys_footprint`, and the system can drop them while the GPU is idle
// and read them back from the file later.
//
// A tensor stays a view only when its file offset is a multiple of its element size; MLX's
// `view(dtype:)` copies otherwise. `saveSafetensorsForMapping` and
// `relayoutSafetensorsForMapping` write files that meet this for every tensor.

/// How a model's weights get into memory.
public enum WeightLoading: Sendable, Hashable {
    /// Read the weights into memory that MLX allocates. They count toward the process's
    /// memory footprint.
    case copy

    /// Map the safetensors files and use the weights in place, without copying them.
    ///
    /// The weights stay clean, file-backed pages that do not count toward the process's
    /// memory footprint (the iOS jetsam limit), and the system can reclaim them while the GPU
    /// is idle. Write files with ``saveSafetensorsForMapping(arrays:metadata:url:)`` or
    /// ``relayoutSafetensorsForMapping(from:to:)`` so no tensor needs a copy. Tensors that
    /// do are copied, and tensors that a model's checkpoint preparation transforms are new
    /// arrays either way.
    case memoryMapped
}

/// Errors from loading or writing memory-mapped safetensors.
public enum MappedSafetensorsError: Error, LocalizedError {
    case cannotOpen(URL, errno: Int32)
    case cannotMap(URL, errno: Int32)
    case malformedHeader(URL, String)
    case unsupportedDtype(URL, tensor: String, dtype: String)
    case tensorTooLarge(URL, tensor: String, bytes: Int)
    case copiedTensors(URL, names: [String])
    case notMetalBacked(URL)

    public var errorDescription: String? {
        switch self {
        case .cannotOpen(let url, let code):
            "Cannot open \(url.path): \(String(cString: strerror(code)))"
        case .cannotMap(let url, let code):
            "Cannot map \(url.path): \(String(cString: strerror(code)))"
        case .malformedHeader(let url, let reason):
            "\(url.lastPathComponent) is not a valid safetensors file: \(reason)"
        case .unsupportedDtype(let url, let tensor, let dtype):
            "\(url.lastPathComponent): tensor \(tensor) has unsupported dtype \(dtype)"
        case .tensorTooLarge(let url, let tensor, let bytes):
            "\(url.lastPathComponent): tensor \(tensor) (\(bytes) bytes) is too large to map"
        case .copiedTensors(let url, let names):
            "\(url.lastPathComponent): \(names.count) tensors are not aligned to their element "
                + "size and would be copied, such as \(names.prefix(3).joined(separator: ", "))"
        case .notMetalBacked(let url):
            "\(url.lastPathComponent): MLX copied the mapping instead of wrapping it"
        }
    }
}

/// The tensors of one memory-mapped safetensors file.
public struct MappedSafetensors {
    /// The tensors, as views into the mapping (or copies, see ``copiedTensorNames``).
    public var arrays: [String: MLXArray]

    /// The file's `__metadata__`.
    public var metadata: [String: String]

    /// Tensors that MLX had to copy because their offset is not a multiple of their
    /// element size.
    public var copiedTensorNames: [String]

    /// Bytes of tensor data used in place from the mapping.
    public var mappedBytes: Int
}

/// Map the safetensors file at `url` and return its tensors without copying them.
///
/// The mapping stays alive as long as any returned array, or any array that is a view of one,
/// is alive. The arrays are evaluated before this returns, so no file I/O happens here; pages
/// are read from the file when the GPU or CPU first uses them.
///
/// - Parameters:
///   - url: a safetensors file
///   - requireNoCopy: throw ``MappedSafetensorsError/copiedTensors(_:names:)`` instead of
///     copying tensors whose offset is not a multiple of their element size
///   - maxWindowBytes: the largest no-copy buffer to create. Metal wires a whole buffer while
///     the GPU uses any part of it, so smaller windows keep unused tensors (an unused vision
///     tower, say) reclaimable. Clamped to the device's maximum buffer length and to 2 GiB.
///   - prefetch: read the returned tensors' pages into the page cache in the background, so
///     the first forward pass does not fault them in one at a time. The pages stay clean.
///   - isRead: tensors to return
public func loadMappedSafetensors(
    url: URL,
    requireNoCopy: Bool = false,
    maxWindowBytes: Int = 512 << 20,
    prefetch: Bool = true,
    isRead: (String) -> Bool = { _ in true }
) throws -> MappedSafetensors {
    let mapping = try SafetensorsMapping(url: url)
    let header = try SafetensorsHeader(base: mapping.base, fileSize: mapping.fileSize, url: url)
    let page = Int(getpagesize())

    let entries = header.tensors.filter { isRead($0.name) }
        .sorted { $0.start < $1.start }

    let misaligned = entries.filter { $0.byteCount > 0 && $0.start % $0.dtype.size != 0 }
        .map(\.name)
    if requireNoCopy, !misaligned.isEmpty {
        throw MappedSafetensorsError.copiedTensors(url, names: misaligned)
    }

    if prefetch, let first = entries.first?.start,
        let end = entries.map({ $0.start + $0.byteCount }).max(), end > first
    {
        mapping.prefetch(from: first / page * page, to: end)
    }

    // MLX shapes are Int32, so a uint8 window must stay under 2 GiB.
    let int32Limit = Int(Int32.max) / page * page
    let deviceLimit = GPU.deviceInfo().maxBufferSize
    var windowLimit = min(max(maxWindowBytes, page), int32Limit)
    if deviceLimit > 0 { windowLimit = min(windowLimit, deviceLimit / page * page) }

    var arrays = [String: MLXArray]()
    var window: (start: Int, length: Int, array: MLXArray)?

    func makeWindow(start: Int, length: Int) throws -> MLXArray {
        let pointer = UnsafeMutableRawPointer(mutating: mapping.base + start)
        let array = MLXArray(rawPointer: pointer, [length], dtype: .uint8) {
            // keeps the mapping alive until MLX releases the buffer
            _ = mapping
        }
        // MLX silently copies when it cannot wrap the pointer in a Metal buffer.
        let wrapped = array.asData(access: .noCopy).data.withUnsafeBytes {
            UnsafeRawPointer($0.baseAddress) == UnsafeRawPointer(pointer)
        }
        guard wrapped else { throw MappedSafetensorsError.notMetalBacked(url) }
        return array
    }

    var mappedBytes = 0
    for entry in entries {
        if entry.byteCount == 0 {
            arrays[entry.name] = MLXArray.zeros(entry.shape, dtype: entry.dtype)
            continue
        }
        let end = entry.start + entry.byteCount
        if window == nil || end > window!.start + window!.length {
            let start = entry.start / page * page
            let span = (end - start + page - 1) / page * page
            guard span <= int32Limit, deviceLimit <= 0 || span <= deviceLimit else {
                throw MappedSafetensorsError.tensorTooLarge(
                    url, tensor: entry.name, bytes: entry.byteCount)
            }
            // A window covers as many later tensors as fit, within the mapping.
            let length = min(max(span, windowLimit), mapping.mappedLength - start)
            window = (start, length, try makeWindow(start: start, length: length))
        }
        let lo = entry.start - window!.start
        let bytes = window!.array[lo ..< (lo + entry.byteCount)]
        arrays[entry.name] =
            entry.dtype == .uint8
            ? bytes.reshaped(entry.shape)
            : bytes.view(dtype: entry.dtype).reshaped(entry.shape)
        if entry.start % entry.dtype.size == 0 { mappedBytes += entry.byteCount }
    }
    eval(Array(arrays.values))

    return MappedSafetensors(
        arrays: arrays, metadata: header.metadata, copiedTensorNames: misaligned,
        mappedBytes: mappedBytes)
}

/// Write `arrays` as a safetensors file that
/// ``loadMappedSafetensors(url:requireNoCopy:maxWindowBytes:prefetch:isRead:)`` can use
/// without copying any tensor.
///
/// The file is ordinary safetensors. The JSON header is padded with spaces, as the format
/// allows, so tensor data starts on a 16 KiB page boundary. Tensors are ordered by descending
/// element size, so each offset is a multiple of its own element size, with no gaps between
/// tensors.
public func saveSafetensorsForMapping(
    arrays: [String: MLXArray], metadata: [String: String] = [:], url: URL
) throws {
    let ordered = arrays.map { (name: $0.key, array: $0.value) }
        .sorted {
            orderedForMapping(($0.name, $0.array.dtype.size), ($1.name, $1.array.dtype.size))
        }
    eval(ordered.map(\.array))

    var tensors = [(name: String, dtype: String, shape: [Int], byteCount: Int)]()
    for (name, array) in ordered {
        guard let dtype = safetensorsDtypeName(array.dtype) else {
            throw MappedSafetensorsError.unsupportedDtype(
                url, tensor: name, dtype: String(describing: array.dtype))
        }
        tensors.append((name, dtype, array.shape, array.nbytes))
    }

    let handle = try createFile(url)
    defer { try? handle.close() }
    try handle.write(contentsOf: mappingHeader(tensors: tensors, metadata: metadata))
    for (_, array) in ordered where array.nbytes > 0 {
        try handle.write(contentsOf: array.asData(access: .noCopyIfContiguous).data)
    }
}

/// Rewrite the safetensors file at `source` into the layout that
/// ``saveSafetensorsForMapping(arrays:metadata:url:)`` writes, copying tensor bytes unchanged.
///
/// Use this for existing checkpoints, such as files written by MLX's own `save`, whose
/// tensors start at arbitrary offsets.
public func relayoutSafetensorsForMapping(from source: URL, to destination: URL) throws {
    let mapping = try SafetensorsMapping(url: source)
    let header = try SafetensorsHeader(
        base: mapping.base, fileSize: mapping.fileSize, url: source, allowUnknownDtypes: true)
    let ordered = header.tensors.sorted {
        orderedForMapping(($0.name, $0.elementSize), ($1.name, $1.elementSize))
    }

    let handle = try createFile(destination)
    defer { try? handle.close() }
    try handle.write(
        contentsOf: mappingHeader(
            tensors: ordered.map { ($0.name, $0.dtypeName, $0.shape, $0.byteCount) },
            metadata: header.metadata))
    for tensor in ordered where tensor.byteCount > 0 {
        let bytes = UnsafeMutableRawPointer(mutating: mapping.base + tensor.start)
        try handle.write(
            contentsOf: Data(bytesNoCopy: bytes, count: tensor.byteCount, deallocator: .none))
    }
}

// MARK: - Implementation

/// A read-only mapping of a whole file, rounded up to whole pages. The tail of the last page
/// reads as zeros.
private final class SafetensorsMapping: @unchecked Sendable {
    let base: UnsafeRawPointer
    let fileSize: Int
    let mappedLength: Int

    init(url: URL) throws {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { throw MappedSafetensorsError.cannotOpen(url, errno: errno) }
        defer { close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0 else {
            throw MappedSafetensorsError.cannotOpen(url, errno: errno)
        }
        let page = Int(getpagesize())
        fileSize = Int(info.st_size)
        mappedLength = max(page, (fileSize + page - 1) / page * page)
        guard let pointer = mmap(nil, mappedLength, PROT_READ, MAP_SHARED, fd, 0),
            pointer != MAP_FAILED
        else { throw MappedSafetensorsError.cannotMap(url, errno: errno) }
        base = UnsafeRawPointer(pointer)
    }

    deinit {
        munmap(UnsafeMutableRawPointer(mutating: base), mappedLength)
    }

    /// Read `start ..< end` into the page cache from several threads, in the background.
    /// `MADV_WILLNEED` blocks until its range is read, and one thread does not reach the
    /// disk's throughput.
    func prefetch(from start: Int, to end: Int) {
        let chunk = 32 << 20
        let chunks = (end - start + chunk - 1) / chunk
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            DispatchQueue.concurrentPerform(iterations: chunks) { index in
                let lo = start + index * chunk
                let pointer = UnsafeMutableRawPointer(mutating: base + lo)
                madvise(pointer, min(chunk, end - lo), MADV_WILLNEED)
            }
        }
    }
}

private struct SafetensorsTensor {
    let name: String
    let dtypeName: String
    let dtype: DType
    let elementSize: Int
    let shape: [Int]
    /// absolute offset in the file
    let start: Int
    let byteCount: Int
}

private struct SafetensorsHeader {
    let tensors: [SafetensorsTensor]
    let metadata: [String: String]

    init(base: UnsafeRawPointer, fileSize: Int, url: URL, allowUnknownDtypes: Bool = false) throws {
        func malformed(_ reason: String) -> MappedSafetensorsError {
            .malformedHeader(url, reason)
        }
        guard fileSize >= 8 else { throw malformed("file is shorter than 8 bytes") }
        let headerLength = Int(UInt64(littleEndian: base.loadUnaligned(as: UInt64.self)))
        // the limit MLX and the reference implementation use
        guard headerLength > 0, headerLength < 100_000_000, 8 + headerLength <= fileSize else {
            throw malformed("header length \(headerLength)")
        }
        let json = Data(
            bytesNoCopy: UnsafeMutableRawPointer(mutating: base + 8),
            count: headerLength, deallocator: .none)
        guard let object = try? JSONSerialization.jsonObject(with: json) as? [String: Any] else {
            throw malformed("header is not a JSON object")
        }
        let dataStart = 8 + headerLength

        var tensors = [SafetensorsTensor]()
        var metadata = [String: String]()
        for (name, value) in object {
            if name == "__metadata__" {
                metadata = (value as? [String: String]) ?? [:]
                continue
            }
            guard let entry = value as? [String: Any],
                let dtypeName = entry["dtype"] as? String,
                let shape = entry["shape"] as? [Int],
                let offsets = entry["data_offsets"] as? [Int], offsets.count == 2
            else { throw malformed("tensor \(name) has no dtype, shape or data_offsets") }

            let dtype = safetensorsDtype(dtypeName)
            guard let elementSize = dtype?.size ?? safetensorsElementSize(dtypeName),
                dtype != nil || allowUnknownDtypes
            else {
                throw MappedSafetensorsError.unsupportedDtype(url, tensor: name, dtype: dtypeName)
            }
            let byteCount = shape.reduce(elementSize, *)
            guard offsets[0] >= 0, offsets[1] - offsets[0] == byteCount,
                dataStart + offsets[1] <= fileSize
            else { throw malformed("tensor \(name) has invalid data_offsets \(offsets)") }

            tensors.append(
                SafetensorsTensor(
                    name: name, dtypeName: dtypeName, dtype: dtype ?? .uint8,
                    elementSize: elementSize, shape: shape, start: dataStart + offsets[0],
                    byteCount: byteCount))
        }
        self.tensors = tensors
        self.metadata = metadata
    }
}

/// MLX's safetensors dtype table (`mlx/io/safetensors.cpp`), including FP8 loaded as bytes.
private func safetensorsDtype(_ name: String) -> DType? {
    switch name {
    case "F32": .float32
    case "F16": .float16
    case "BF16": .bfloat16
    case "I64": .int64
    case "I32": .int32
    case "I16": .int16
    case "I8": .int8
    case "U64": .uint64
    case "U32": .uint32
    case "U16": .uint16
    case "U8", "F8_E4M3", "F8_E8M0": .uint8
    case "BOOL": .bool
    case "C64": .complex64
    default: nil
    }
}

/// Element sizes of safetensors dtypes MLX does not load, for relayout only.
private func safetensorsElementSize(_ name: String) -> Int? {
    switch name {
    case "F64": 8
    case "F8_E5M2": 1
    default: nil
    }
}

private func safetensorsDtypeName(_ dtype: DType) -> String? {
    switch dtype {
    case .float32: "F32"
    case .float16: "F16"
    case .bfloat16: "BF16"
    case .int64: "I64"
    case .int32: "I32"
    case .int16: "I16"
    case .int8: "I8"
    case .uint64: "U64"
    case .uint32: "U32"
    case .uint16: "U16"
    case .uint8: "U8"
    case .bool: "BOOL"
    case .complex64: "C64"
    default: nil
    }
}

/// Larger elements first, so every offset is a multiple of its element size; then by name.
private func orderedForMapping(_ a: (String, Int), _ b: (String, Int)) -> Bool {
    a.1 != b.1 ? a.1 > b.1 : a.0 < b.0
}

/// The 8-byte length and JSON header, padded with spaces so tensor data starts on a page.
private func mappingHeader(
    tensors: [(name: String, dtype: String, shape: [Int], byteCount: Int)],
    metadata: [String: String]
) throws -> Data {
    var object = [String: Any]()
    if !metadata.isEmpty { object["__metadata__"] = metadata }
    var offset = 0
    for tensor in tensors {
        object[tensor.name] = [
            "dtype": tensor.dtype, "shape": tensor.shape,
            "data_offsets": [offset, offset + tensor.byteCount],
        ]
        offset += tensor.byteCount
    }
    var json = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    let alignment = safetensorsMappingAlignment
    let padded = (8 + json.count + alignment - 1) / alignment * alignment - 8
    json.append(Data(repeating: 0x20, count: padded - json.count))

    var header = Data()
    withUnsafeBytes(of: UInt64(json.count).littleEndian) { header.append(contentsOf: $0) }
    header.append(json)
    return header
}

/// Tensor data starts at a multiple of this, the arm64 page size on iOS and macOS.
let safetensorsMappingAlignment = 16384

private func createFile(_ url: URL) throws -> FileHandle {
    guard FileManager.default.createFile(atPath: url.path, contents: nil) else {
        throw MappedSafetensorsError.cannotOpen(url, errno: errno)
    }
    return try FileHandle(forWritingTo: url)
}
