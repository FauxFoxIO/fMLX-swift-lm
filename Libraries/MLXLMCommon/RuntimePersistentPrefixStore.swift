// Copyright © 2026 Faux Fox.

import CryptoKit
import Darwin
import Foundation
import MLX

/// Optional disk storage for evaluated text prefixes.
public struct RuntimePersistentCacheConfiguration: Sendable {
    /// A dedicated directory. Symbolic links in this path are rejected.
    public let directory: URL
    /// Bounds committed tensor and manifest bytes in each identity/layout namespace.
    /// Atomic publication may additionally use one manifest, limited to 4 MiB.
    public let maximumBytes: Int

    public init(directory: URL, maximumBytes: Int) {
        self.directory = directory
        self.maximumBytes = maximumBytes
    }
}

internal enum RuntimePersistentPrefixStoreError: Error {
    case invalidConfiguration
    case unsafePath
    case unsupportedCache
    case invalidSnapshot
    case entryTooLarge
    case io(Int32)
}

/// Synchronous and actor-owned. Directory locks also serialize other instances/processes.
internal final class RuntimePersistentPrefixStore {
    private typealias Failure = RuntimePersistentPrefixStoreError
    private static let version = 1
    private static let manifestLimit = 4 * 1024 * 1024
    private let configuration: RuntimePersistentCacheConfiguration
    private let identity: PrefixCacheIdentity
    private let layoutFingerprint: String
    private let directoryHandle: DirectoryHandle
    private var directory: Int32 { directoryHandle.descriptor }
    private var entries: [Entry] = []

    private final class DirectoryHandle {
        let descriptor: Int32
        init(_ descriptor: Int32) { self.descriptor = descriptor }
        deinit { close(descriptor) }
    }

    var storedBytes: Int {
        entries.reduce(0) {
            let sum = $0.addingReportingOverflow($1.byteCount)
            return sum.overflow ? Int.max : sum.partialValue
        }
    }
    var entryCount: Int { entries.count }

    private struct Namespace: Codable {
        let version: Int
        let identity: PrefixCacheIdentity
        let layout: String
    }

    private struct Tensor: Codable, Equatable {
        let shape: [Int]
        let dtype: String
        let byteCount: Int
    }

    private struct Layer: Codable {
        let kind: String
        let offset: Int
        let step: Int?
        let metadata: [String]
        let tensors: [Tensor]
    }

    private struct Manifest: Codable {
        let version: Int
        let identity: PrefixCacheIdentity
        let layout: String
        let tokens: [Int]
        let layers: [Layer]
        let dataFile: String
        let dataBytes: Int
        let dataDigest: String
        var access: UInt64
    }

    private struct Envelope: Codable {
        let payload: Data
        let digest: String
    }

    private struct Entry {
        let name: String
        var manifest: Manifest
        let manifestBytes: Int
        var byteCount: Int { manifestBytes + manifest.dataBytes }
    }

    init(
        configuration: RuntimePersistentCacheConfiguration, identity: PrefixCacheIdentity,
        layoutFingerprint: String
    ) throws {
        guard configuration.directory.isFileURL, configuration.maximumBytes > 0,
            !layoutFingerprint.isEmpty
        else { throw Failure.invalidConfiguration }
        self.configuration = configuration
        self.identity = identity
        self.layoutFingerprint = layoutFingerprint
        let namespace = try Self.encode(
            Namespace(version: Self.version, identity: identity, layout: layoutFingerprint))
        let root = try Self.openDirectory(configuration.directory)
        defer { close(root) }
        let name = "prefix-v1-" + Self.digest(namespace)
        if mkdirat(root, name, 0o700) != 0 && errno != EEXIST { throw Failure.io(errno) }
        let descriptor = openat(root, name, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
        guard descriptor >= 0 else { throw Failure.unsafePath }
        directoryHandle = DirectoryHandle(descriptor)
        try locked {
            let marker = "owner-v1.json"
            if let existing = try? read(marker, limit: Self.manifestLimit) {
                guard existing == namespace else { throw Failure.unsafePath }
            } else {
                guard try names().isEmpty else { throw Failure.unsafePath }
                try writeAtomically(namespace, to: marker)
            }
            try recover()
        }
    }

    func store(tokens: [Int], cache: [KVCache]) throws {
        guard !tokens.isEmpty, tokens.allSatisfy({ $0 >= 0 }), !cache.isEmpty,
            cache.count <= 4096
        else { throw Failure.invalidSnapshot }
        var bytes = Data()
        var layers: [Layer] = []
        for cache in cache {
            let kind = try Self.kind(cache)
            guard cache.offset >= 0, kind == "mamba" || cache.offset == tokens.count else {
                throw Failure.invalidSnapshot
            }
            let raw = cache.innerState()
            if kind != "mamba" {
                guard
                    raw.allSatisfy({
                        $0.ndim == 4 && $0.dim(2) >= cache.offset
                    })
                else { throw Failure.invalidSnapshot }
            }
            let arrays = cache.state
            let layer = Layer(
                kind: kind, offset: cache.offset, step: (cache as? KVCacheSimple)?.step,
                metadata: cache.metaState,
                tensors: arrays.map {
                    Tensor(
                        shape: $0.shape, dtype: String(describing: $0.dtype), byteCount: $0.nbytes)
                })
            try validate(layer, tokenCount: tokens.count)
            for array in arrays {
                guard array.nbytes <= configuration.maximumBytes - bytes.count else {
                    throw Failure.entryTooLarge
                }
                // The copied, evaluated bytes cannot retain a mutable cache or lazy graph.
                bytes.append(array.asData(access: .copy).data)
            }
            layers.append(layer)
        }
        try locked {
            try recover()
            let name = try entryName(tokens)
            let manifest = Manifest(
                version: Self.version, identity: identity, layout: layoutFingerprint,
                tokens: tokens, layers: layers,
                dataFile: "data-v1-\(UUID().uuidString.lowercased()).bin",
                dataBytes: bytes.count, dataDigest: Self.digest(bytes), access: nextAccess())
            let manifestData = try envelope(manifest)
            guard manifestData.count <= min(Self.manifestLimit, configuration.maximumBytes),
                bytes.count <= configuration.maximumBytes - manifestData.count
            else { throw Failure.entryTooLarge }
            // Eviction precedes publication so tensor staging stays within the disk budget.
            if let previous = entries.first(where: { $0.name == name }) {
                try remove(previous)
            }
            let required = bytes.count + manifestData.count
            while storedBytes > configuration.maximumBytes - required,
                let oldest = entries.min(by: Self.older)
            {
                try remove(oldest)
            }
            do {
                try writeAtomically(bytes, to: manifest.dataFile)
                try writeAtomically(manifestData, to: name)
                entries.append(
                    Entry(name: name, manifest: manifest, manifestBytes: manifestData.count))
            } catch {
                try? recover()
                throw error
            }
        }
    }

    func restore(prompt: [Int], maximumPrefixTokens: Int, prototype: [KVCache]) throws -> (
        tokens: [Int], cache: [KVCache]
    )? {
        guard maximumPrefixTokens > 0 else { return nil }
        for cache in prototype { _ = try Self.kind(cache) }
        return try locked {
            try recover()
            let candidates = entries.filter {
                $0.manifest.tokens.count <= maximumPrefixTokens
                    && prompt.starts(with: $0.manifest.tokens)
            }.sorted { $0.manifest.tokens.count > $1.manifest.tokens.count }
            for entry in candidates {
                do {
                    try validatePrototype(prototype, layers: entry.manifest.layers)
                    let bytes = try read(entry.manifest.dataFile, limit: entry.manifest.dataBytes)
                    guard bytes.count == entry.manifest.dataBytes,
                        Self.digest(bytes) == entry.manifest.dataDigest
                    else { throw Failure.invalidSnapshot }
                    let restored = try reconstruct(entry.manifest.layers, bytes: bytes)
                    var touched = entry.manifest
                    touched.access = nextAccess()
                    let manifestData = try envelope(touched)
                    try writeAtomically(manifestData, to: entry.name)
                    entries.removeAll { $0.name == entry.name }
                    entries.append(
                        Entry(
                            name: entry.name, manifest: touched, manifestBytes: manifestData.count))
                    while storedBytes > configuration.maximumBytes,
                        let oldest = entries.min(by: Self.older)
                    {
                        try remove(oldest)
                    }
                    return (entry.manifest.tokens, restored)
                } catch {
                    // A failed candidate must never leak a partially restored cache.
                    try remove(entry)
                }
            }
            return nil
        }
    }

    func clear() throws {
        try locked {
            try recover()
            for entry in entries { try remove(entry) }
        }
    }

    private static func kind(_ cache: KVCache) throws -> String {
        if type(of: cache) == KVCacheSimple.self { return "simple" }
        if type(of: cache) == MambaCache.self { return "mamba" }
        if type(of: cache) == QuantizedKVCache.self,
            let quantized = cache as? QuantizedKVCache, quantized.mode == .affine
        {
            return "quantized"
        }
        throw Failure.unsupportedCache
    }

    private func validate(_ layer: Layer, tokenCount: Int) throws {
        // Recurrent offsets follow the model's convention; attention offsets count tokens.
        guard layer.offset >= 0, layer.kind == "mamba" || layer.offset == tokenCount,
            tokenCount > 0, !layer.tensors.isEmpty, layer.tensors.count <= 6
        else { throw Failure.invalidSnapshot }
        for tensor in layer.tensors {
            guard !tensor.shape.isEmpty, tensor.shape.count <= 8,
                tensor.shape.allSatisfy({ $0 > 0 && $0 <= Int(Int32.max) }),
                let dtype = Self.dtype(tensor.dtype), tensor.byteCount > 0
            else { throw Failure.invalidSnapshot }
            var count = dtype.size
            for dimension in tensor.shape {
                guard count <= configuration.maximumBytes / dimension else {
                    throw Failure.invalidSnapshot
                }
                count *= dimension
            }
            guard count == tensor.byteCount else { throw Failure.invalidSnapshot }
        }
        switch layer.kind {
        case "simple":
            guard layer.tensors.count == 2, layer.metadata == [""],
                let step = layer.step, step > 0, step <= Int(Int32.max)
            else { throw Failure.invalidSnapshot }
            try validateAttention(layer)
            guard layer.tensors.allSatisfy({ Self.isFloat($0.dtype) }) else {
                throw Failure.invalidSnapshot
            }
        case "quantized":
            guard layer.tensors.count == 6, layer.step == nil,
                layer.metadata.count == 4, layer.metadata[0] == "256",
                Int(layer.metadata[1]) == tokenCount,
                let group = Int(layer.metadata[2]), [32, 64, 128].contains(group),
                let bits = Int(layer.metadata[3]), [2, 3, 4, 5, 6, 8].contains(bits)
            else { throw Failure.invalidSnapshot }
            try validateAttention(layer)
            for start in [0, 3] {
                let weight = layer.tensors[start]
                let scale = layer.tensors[start + 1]
                let bias = layer.tensors[start + 2]
                guard weight.dtype == "uint32", Self.isFloat(scale.dtype),
                    scale == bias,
                    weight.shape[3] * 32 == scale.shape[3] * group * bits
                else { throw Failure.invalidSnapshot }
            }
        case "mamba":
            guard layer.step == nil, (2 ... 4).contains(layer.metadata.count),
                layer.metadata[0] == "2"
            else { throw Failure.invalidSnapshot }
            let slots = layer.metadata[1].split(separator: ",").compactMap { Int($0) }
            guard slots.count == layer.tensors.count,
                slots == Array(Set(slots)).sorted(), slots.allSatisfy({ (0 ... 1).contains($0) }),
                slots.map(String.init).joined(separator: ",") == layer.metadata[1],
                layer.tensors.allSatisfy({ $0.shape[0] == 1 && Self.isFloat($0.dtype) })
            else { throw Failure.invalidSnapshot }
            for metadata in layer.metadata.dropFirst(2) where !metadata.isEmpty {
                guard let value = Int32(metadata), String(value) == metadata else {
                    throw Failure.invalidSnapshot
                }
            }
        default: throw Failure.unsupportedCache
        }
    }

    private func validateAttention(_ layer: Layer) throws {
        guard
            layer.tensors.allSatisfy({
                $0.shape.count == 4 && $0.shape[0] == 1 && $0.shape[2] == layer.offset
                    && $0.shape[1] == layer.tensors[0].shape[1]
            })
        else { throw Failure.invalidSnapshot }
    }

    private func validatePrototype(_ prototype: [KVCache], layers: [Layer]) throws {
        guard prototype.count == layers.count else { throw Failure.invalidSnapshot }
        for (cache, layer) in zip(prototype, layers) {
            let kind = try Self.kind(cache)
            guard kind == layer.kind || (kind == "simple" && layer.kind == "quantized") else {
                throw Failure.invalidSnapshot
            }
            if let quantized = cache as? QuantizedKVCache {
                guard Int(layer.metadata[2]) == quantized.groupSize,
                    Int(layer.metadata[3]) == quantized.bits
                else { throw Failure.invalidSnapshot }
            }
            let tensors = cache.innerState()
            if !tensors.isEmpty && kind == layer.kind {
                guard tensors.count == layer.tensors.count else { throw Failure.invalidSnapshot }
                for (actual, stored) in zip(tensors, layer.tensors) {
                    guard String(describing: actual.dtype) == stored.dtype,
                        actual.ndim == stored.shape.count,
                        actual.shape.enumerated().allSatisfy({ index, size in
                            (layer.kind != "mamba" && index == 2) || size == stored.shape[index]
                        })
                    else { throw Failure.invalidSnapshot }
                }
                if let mamba = cache as? MambaCache {
                    guard mamba.metaState[1] == layer.metadata[1] else {
                        throw Failure.invalidSnapshot
                    }
                }
            }
        }
    }

    private func reconstruct(_ layers: [Layer], bytes: Data) throws -> [KVCache] {
        var position = 0
        return try layers.map { layer in
            var arrays: [MLXArray] = []
            for tensor in layer.tensors {
                guard let dtype = Self.dtype(tensor.dtype),
                    tensor.byteCount <= bytes.count - position
                else { throw Failure.invalidSnapshot }
                let end = position + tensor.byteCount
                arrays.append(
                    MLXArray(bytes.subdata(in: position ..< end), tensor.shape, dtype: dtype))
                position = end
            }
            let cache: BaseKVCache
            switch layer.kind {
            case "simple":
                let simple = KVCacheSimple()
                simple.step = layer.step ?? 256
                simple.state = arrays
                cache = simple
            case "quantized":
                let quantized = QuantizedKVCache()
                quantized.state = arrays
                quantized.metaState = layer.metadata
                cache = quantized
            case "mamba":
                let mamba = MambaCache()
                mamba.restoreFromMetaState(state: arrays, savedMetaState: layer.metadata)
                cache = mamba
            default: throw Failure.unsupportedCache
            }
            cache.offset = layer.offset
            eval(cache.state)
            return cache
        }
    }

    private func recover() throws {
        entries = []
        let allNames = try names()
        for name in allNames where Self.isManifest(name) {
            do {
                let data = try read(
                    name, limit: min(configuration.maximumBytes, Self.manifestLimit))
                let envelope = try JSONDecoder().decode(Envelope.self, from: data)
                guard Self.digest(envelope.payload) == envelope.digest else {
                    throw Failure.invalidSnapshot
                }
                let manifest = try JSONDecoder().decode(Manifest.self, from: envelope.payload)
                guard manifest.version == Self.version, manifest.identity == identity,
                    manifest.layout == layoutFingerprint, !manifest.tokens.isEmpty,
                    manifest.tokens.allSatisfy({ $0 >= 0 }),
                    try entryName(manifest.tokens) == name,
                    !manifest.layers.isEmpty, manifest.layers.count <= 4096,
                    Self.isData(manifest.dataFile), manifest.dataBytes > 0,
                    manifest.dataBytes <= configuration.maximumBytes - data.count,
                    Self.isDigest(manifest.dataDigest)
                else { throw Failure.invalidSnapshot }
                var count = 0
                for layer in manifest.layers {
                    try validate(layer, tokenCount: manifest.tokens.count)
                    for tensor in layer.tensors {
                        guard tensor.byteCount <= manifest.dataBytes - count else {
                            throw Failure.invalidSnapshot
                        }
                        count += tensor.byteCount
                    }
                }
                guard count == manifest.dataBytes,
                    try fileSize(manifest.dataFile) == manifest.dataBytes
                else { throw Failure.invalidSnapshot }
                entries.append(Entry(name: name, manifest: manifest, manifestBytes: data.count))
            } catch {
                try delete(name)
            }
        }
        let referenced = Set(entries.map { $0.manifest.dataFile })
        for name in allNames
        where (Self.isData(name) && !referenced.contains(name)) || Self.isTemporary(name) {
            try delete(name)
        }
        while storedBytes > configuration.maximumBytes, let oldest = entries.min(by: Self.older) {
            try remove(oldest)
        }
    }

    private func remove(_ entry: Entry) throws {
        try delete(entry.name)
        entries.removeAll { $0.name == entry.name }
        if !entries.contains(where: { $0.manifest.dataFile == entry.manifest.dataFile }) {
            try delete(entry.manifest.dataFile)
        }
    }

    private func nextAccess() -> UInt64 {
        let latest = entries.map { $0.manifest.access }.max() ?? 0
        return latest == UInt64.max ? latest : latest + 1
    }

    private static func older(_ lhs: Entry, _ rhs: Entry) -> Bool {
        if lhs.manifest.access != rhs.manifest.access {
            return lhs.manifest.access < rhs.manifest.access
        }
        return lhs.name < rhs.name
    }

    private func entryName(_ tokens: [Int]) throws -> String {
        "entry-v1-" + Self.digest(try Self.encode(tokens)) + ".json"
    }

    private func envelope(_ manifest: Manifest) throws -> Data {
        let payload = try Self.encode(manifest)
        return try Self.encode(Envelope(payload: payload, digest: Self.digest(payload)))
    }

    private static func encode(_ value: some Encodable) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(value)
    }

    private static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func dtype(_ name: String) -> DType? {
        DType.allCases.first { String(describing: $0) == name }
    }

    private static func isFloat(_ name: String) -> Bool {
        ["float16", "bfloat16", "float32"].contains(name)
    }

    private static func isDigest(_ name: String) -> Bool {
        name.count == 64
            && name.utf8.allSatisfy { (48 ... 57).contains($0) || (97 ... 102).contains($0) }
    }

    private static func isManifest(_ name: String) -> Bool {
        name.hasPrefix("entry-v1-") && name.hasSuffix(".json")
            && isDigest(String(name.dropFirst(9).dropLast(5)))
    }

    private static func isData(_ name: String) -> Bool {
        name.hasPrefix("data-v1-") && name.hasSuffix(".bin")
            && UUID(uuidString: String(name.dropFirst(8).dropLast(4))) != nil
    }

    private static func isTemporary(_ name: String) -> Bool {
        name.hasPrefix("temp-v1-") && name.hasSuffix(".tmp")
            && UUID(uuidString: String(name.dropFirst(8).dropLast(4))) != nil
    }

    private func locked<T>(_ body: () throws -> T) throws -> T {
        guard flock(directory, LOCK_EX) == 0 else { throw Failure.io(errno) }
        defer { flock(directory, LOCK_UN) }
        return try body()
    }

    private static func openDirectory(_ url: URL) throws -> Int32 {
        guard url.path.hasPrefix("/"), !url.pathComponents.contains("..") else {
            throw Failure.unsafePath
        }
        var descriptor = open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        guard descriptor >= 0 else { throw Failure.io(errno) }
        do {
            for component in url.pathComponents.dropFirst() where component != "." {
                if mkdirat(descriptor, component, 0o700) != 0 && errno != EEXIST {
                    throw Failure.io(errno)
                }
                let next = openat(
                    descriptor, component, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
                guard next >= 0 else { throw Failure.unsafePath }
                close(descriptor)
                descriptor = next
            }
            return descriptor
        } catch {
            close(descriptor)
            throw error
        }
    }

    private func names() throws -> [String] {
        let descriptor = openat(directory, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        guard descriptor >= 0 else { throw Failure.io(errno) }
        guard let stream = fdopendir(descriptor) else {
            close(descriptor)
            throw Failure.io(errno)
        }
        defer { closedir(stream) }
        var result: [String] = []
        errno = 0
        while let entry = readdir(stream) {
            let name = withUnsafePointer(to: &entry.pointee.d_name) {
                $0.withMemoryRebound(to: CChar.self, capacity: Int(entry.pointee.d_namlen) + 1) {
                    String(cString: $0)
                }
            }
            if name != "." && name != ".." && name != "owner-v1.json" { result.append(name) }
            errno = 0
        }
        guard errno == 0 else { throw Failure.io(errno) }
        return result
    }

    private func openFile(_ name: String) throws -> (Int32, Int) {
        guard !name.contains("/"), name != ".", name != ".." else { throw Failure.unsafePath }
        let descriptor = openat(directory, name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
        guard descriptor >= 0 else { throw Failure.io(errno) }
        var info = stat()
        guard fstat(descriptor, &info) == 0, info.st_mode & S_IFMT == S_IFREG,
            info.st_nlink == 1, info.st_size >= 0, info.st_size <= Int64(Int.max)
        else {
            close(descriptor)
            throw Failure.unsafePath
        }
        return (descriptor, Int(info.st_size))
    }

    private func fileSize(_ name: String) throws -> Int {
        let (descriptor, size) = try openFile(name)
        defer { close(descriptor) }
        return size
    }

    private func read(_ name: String, limit: Int) throws -> Data {
        let (descriptor, size) = try openFile(name)
        defer { close(descriptor) }
        guard size <= limit else { throw Failure.entryTooLarge }
        var data = Data(count: size)
        try data.withUnsafeMutableBytes { buffer in
            var position = 0
            while position < size {
                let count = Darwin.read(
                    descriptor, buffer.baseAddress!.advanced(by: position), size - position)
                if count < 0 && errno == EINTR { continue }
                guard count > 0 else { throw Failure.invalidSnapshot }
                position += count
            }
        }
        return data
    }

    private func writeAtomically(_ data: Data, to name: String) throws {
        let temporary = "temp-v1-\(UUID().uuidString.lowercased()).tmp"
        let descriptor = openat(
            directory, temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard descriptor >= 0 else { throw Failure.io(errno) }
        defer {
            close(descriptor)
            unlinkat(directory, temporary, 0)
        }
        try data.withUnsafeBytes { buffer in
            var position = 0
            while position < data.count {
                let count = Darwin.write(
                    descriptor, buffer.baseAddress!.advanced(by: position), data.count - position)
                if count < 0 && errno == EINTR { continue }
                guard count > 0 else { throw Failure.io(errno) }
                position += count
            }
        }
        guard fsync(descriptor) == 0,
            renameat(directory, temporary, directory, name) == 0,
            fsync(directory) == 0
        else { throw Failure.io(errno) }
    }

    private func delete(_ name: String) throws {
        // Unlink only our namespace's flat filenames; never follow links or recurse.
        guard Self.isManifest(name) || Self.isData(name) || Self.isTemporary(name) else {
            throw Failure.unsafePath
        }
        if unlinkat(directory, name, 0) != 0 && errno != ENOENT { throw Failure.io(errno) }
    }
}
