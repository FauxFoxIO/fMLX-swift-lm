// Copyright © 2026 Faux Fox.

import CryptoKit
import Darwin
import Foundation
import MLX
import XCTest

@testable import MLXLMCommon

final class RuntimePersistentPrefixStoreTests: XCTestCase {
    private let identity = PrefixCacheIdentity(
        modelRevision: "weights-1", tokenizerRevision: "tokens-1",
        chatTemplateRevision: "chat-1", adapterRevision: "none", cacheLayoutRevision: "layout-1")

    private func withDirectory(_ body: (URL) throws -> Void) throws {
        // Foundation keeps macOS's /var alias even after resolvingSymlinksInPath().
        let canonical = try XCTUnwrap(
            FileManager.default.temporaryDirectory.withUnsafeFileSystemRepresentation { path in
                path.flatMap { realpath($0, nil) }
            })
        defer { free(canonical) }
        let directory = URL(fileURLWithPath: String(cString: canonical), isDirectory: true)
            .appendingPathComponent("runtime-prefix-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try body(directory)
    }

    private func store(
        _ directory: URL, maximumBytes: Int = 1_048_576, layout: String = "simple-f32-head4",
        identity: PrefixCacheIdentity? = nil
    ) throws -> RuntimePersistentPrefixStore {
        try RuntimePersistentPrefixStore(
            configuration: .init(directory: directory, maximumBytes: maximumBytes),
            identity: identity ?? self.identity, layoutFingerprint: layout)
    }

    private func cache(_ tokens: [Int], width: Int = 4) -> KVCacheSimple {
        let cache = KVCacheSimple()
        let values = MLXArray(tokens.flatMap { Array(repeating: Float($0), count: width) })
            .reshaped(1, 1, tokens.count, width)
        cache.state = [values, values + 1]
        return cache
    }

    private func files(_ root: URL, prefix: String) throws -> [URL] {
        let namespace = try XCTUnwrap(
            FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
                .first { $0.lastPathComponent.hasPrefix("prefix-v1-") })
        return try FileManager.default.contentsOfDirectory(
            at: namespace, includingPropertiesForKeys: nil
        ).filter { $0.lastPathComponent.hasPrefix(prefix) }
    }

    private func editManifest(
        _ root: URL, resign: Bool = true, _ edit: (inout [String: Any]) throws -> Void
    ) throws {
        let file = try XCTUnwrap(files(root, prefix: "entry-v1-").first)
        var envelope = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
        let payload = try XCTUnwrap(
            Data(base64Encoded: try XCTUnwrap(envelope["payload"] as? String)))
        var manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: payload) as? [String: Any])
        try edit(&manifest)
        let changed = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys])
        envelope["payload"] = changed.base64EncodedString()
        if resign { envelope["digest"] = Self.digest(changed) }
        try JSONSerialization.data(withJSONObject: envelope).write(to: file)
    }

    private static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    func testRestartRecoversLongestExactPrefixAndHonorsLimit() throws {
        try withDirectory { directory in
            do {
                let writer = try store(directory)
                try writer.store(tokens: [1, 2], cache: [cache([1, 2])])
                try writer.store(tokens: [1, 2, 3], cache: [cache([1, 2, 3])])
            }
            let reader = try store(directory)
            XCTAssertEqual(reader.entryCount, 2)
            let longest = try XCTUnwrap(
                reader.restore(
                    prompt: [1, 2, 3, 4], maximumPrefixTokens: 3, prototype: [KVCacheSimple()]))
            XCTAssertEqual(longest.tokens, [1, 2, 3])
            XCTAssertEqual(longest.cache[0].offset, 3)
            XCTAssertEqual(
                longest.cache[0].state[0].asArray(Float.self),
                cache([1, 2, 3]).state[0].asArray(Float.self))
            XCTAssertEqual(
                try reader.restore(
                    prompt: [1, 2, 3], maximumPrefixTokens: 2, prototype: [KVCacheSimple()])?
                    .tokens,
                [1, 2])
            XCTAssertNil(
                try reader.restore(
                    prompt: [1, 9, 3], maximumPrefixTokens: 3, prototype: [KVCacheSimple()]))
            XCTAssertNil(
                try reader.restore(
                    prompt: [1], maximumPrefixTokens: 3, prototype: [KVCacheSimple()]))
            XCTAssertNil(
                try reader.restore(
                    prompt: [1, 2], maximumPrefixTokens: 0, prototype: [KVCacheSimple()]))
        }
    }

    func testEvaluatedSnapshotAndRestoredBranchesRemainIndependent() throws {
        try withDirectory { directory in
            let disk = try store(directory)
            let original = cache([3, 4])
            try disk.store(tokens: [3, 4], cache: [original])
            original.state[0][.ellipsis] = MLXArray.zeros([1, 1, 2, 4])
            let first = try XCTUnwrap(
                disk.restore(
                    prompt: [3, 4, 5], maximumPrefixTokens: 2, prototype: [KVCacheSimple()]))
            let expected: [Float] = [3, 3, 3, 3, 4, 4, 4, 4]
            XCTAssertEqual(first.cache[0].state[0].asArray(Float.self), expected)
            first.cache[0].state[0][.ellipsis] = MLXArray.ones([1, 1, 2, 4]) * 99
            let second = try XCTUnwrap(
                disk.restore(
                    prompt: [3, 4, 6], maximumPrefixTokens: 2, prototype: [KVCacheSimple()]))
            XCTAssertFalse((first.cache[0] as AnyObject) === (second.cache[0] as AnyObject))
            XCTAssertEqual(second.cache[0].state[0].asArray(Float.self), expected)
            let added = MLXArray.ones([1, 1, 1, 4])
            _ = first.cache[0].update(keys: added, values: added)
            XCTAssertEqual(first.cache[0].offset, 3)
            XCTAssertEqual(second.cache[0].offset, 2)
        }
    }

    func testRecurrentSparseSlotsMetadataAndOffsetsSurviveRestart() throws {
        try withDirectory { directory in
            let recurrent = MambaCache(leftPadding: [-2])
            recurrent[1] = MLXArray.ones([1, 2, 3]) * 7
            recurrent.lengths = MLXArray([-1])
            recurrent.offset = 7
            do {
                try store(directory, layout: "mamba").store(tokens: [1, 2, 3], cache: [recurrent])
            }
            let reader = try store(directory, layout: "mamba")
            let first = try XCTUnwrap(
                reader.restore(
                    prompt: [1, 2, 3, 4], maximumPrefixTokens: 3, prototype: [MambaCache()]))
            let restored = try XCTUnwrap(first.cache.first as? MambaCache)
            XCTAssertEqual(restored.offset, 7)
            XCTAssertNil(restored[0])
            XCTAssertEqual(restored[1]?.asArray(Float.self), Array(repeating: 7, count: 6))
            XCTAssertEqual(restored.leftPadding?.asArray(Int.self), [-2])
            XCTAssertEqual(restored.lengths?.asArray(Int.self), [-1])
            restored[1] = restored[1]! + 20
            restored.offset += 1
            let next = try XCTUnwrap(
                reader.restore(
                    prompt: [1, 2, 3, 5], maximumPrefixTokens: 3, prototype: [MambaCache()]))
            XCTAssertEqual(next.cache[0].offset, 7)
            XCTAssertEqual(
                (next.cache[0] as? MambaCache)?[1]?.asArray(Float.self),
                Array(repeating: 7, count: 6))
        }
    }

    func testHybridRecurrentOffsetRemainsNativeAfterAdvanceAndRestart() throws {
        try withDirectory { directory in
            let tokens = [1, 2, 3]
            let recurrent = MambaCache(leftPadding: [1])
            recurrent.prepare(lengths: [3])
            recurrent[0] = MLXArray.zeros([1, 2, 4])
            recurrent[1] = MLXArray.ones([1, 2, 2])
            recurrent.advance(tokens.count)
            XCTAssertEqual(recurrent.offset, 0)
            do {
                try store(directory, layout: "hybrid").store(
                    tokens: tokens, cache: [cache(tokens), recurrent])
            }
            let reader = try store(directory, layout: "hybrid")
            let hit = try XCTUnwrap(
                reader.restore(
                    prompt: tokens + [4], maximumPrefixTokens: 3,
                    prototype: [KVCacheSimple(), MambaCache()]))
            XCTAssertEqual(hit.tokens, tokens)
            XCTAssertEqual(hit.cache.map { $0.offset }, [3, 0])
            let restored = try XCTUnwrap(hit.cache[1] as? MambaCache)
            XCTAssertEqual(restored.leftPadding?.asArray(Int.self), [-2])
            XCTAssertEqual(restored.lengths?.asArray(Int.self), [0])
            XCTAssertEqual(restored[0]?.asArray(Float.self), Array(repeating: 0, count: 8))
            XCTAssertEqual(restored[1]?.asArray(Float.self), Array(repeating: 1, count: 4))
            restored.advance(1)
            XCTAssertEqual(restored.offset, 0)
            XCTAssertEqual(restored.lengths?.asArray(Int.self), [-1])
            XCTAssertEqual(recurrent.lengths?.asArray(Int.self), [0])
        }
    }

    func testAffineQuantizedCacheRestoresAgainstSimplePrototype() throws {
        try withDirectory { directory in
            let quantized = try cache([1, 2, 3], width: 64).toQuantized(groupSize: 32, bits: 4)
            let disk = try store(directory, layout: "simple-quantized-affine-4-32")
            try disk.store(tokens: [1, 2, 3], cache: [quantized])
            let hit = try XCTUnwrap(
                disk.restore(
                    prompt: [1, 2, 3, 4], maximumPrefixTokens: 3, prototype: [KVCacheSimple()]))
            let restored = try XCTUnwrap(hit.cache[0] as? QuantizedKVCache)
            XCTAssertEqual(restored.groupSize, 32)
            XCTAssertEqual(restored.bits, 4)
            XCTAssertEqual(restored.offset, 3)
            for (lhs, rhs) in zip(restored.state, quantized.state) {
                XCTAssertEqual(lhs.dtype, rhs.dtype)
                XCTAssertEqual(lhs.shape, rhs.shape)
                XCTAssertEqual(lhs.asData(access: .copy).data, rhs.asData(access: .copy).data)
            }
        }
    }

    func testIdentityAndActualLayoutNamespacesNeverCross() throws {
        try withDirectory { directory in
            let original = try store(directory)
            try original.store(tokens: [1, 2], cache: [cache([1, 2])])
            let changedIdentity = PrefixCacheIdentity(
                modelRevision: "weights-2", tokenizerRevision: "tokens-1",
                chatTemplateRevision: "chat-1",
                adapterRevision: "none", cacheLayoutRevision: "layout-1")
            XCTAssertNil(
                try store(directory, identity: changedIdentity).restore(
                    prompt: [1, 2, 3], maximumPrefixTokens: 2, prototype: [KVCacheSimple()]))
            XCTAssertNil(
                try store(directory, layout: "different-actual-layout").restore(
                    prompt: [1, 2, 3], maximumPrefixTokens: 2, prototype: [KVCacheSimple()]))
            XCTAssertNotNil(
                try original.restore(
                    prompt: [1, 2, 3], maximumPrefixTokens: 2, prototype: [KVCacheSimple()]))
        }
    }

    func testDigestCorruptionTruncationAndOversizedFilesAreMisses() throws {
        for mutation in 0 ..< 5 {
            try withDirectory { directory in
                let disk = try store(directory, maximumBytes: 4096)
                try disk.store(tokens: [1, 2], cache: [cache([1, 2])])
                let dataFile = try XCTUnwrap(files(directory, prefix: "data-v1-").first)
                let manifestFile = try XCTUnwrap(files(directory, prefix: "entry-v1-").first)
                switch mutation {
                case 0:
                    var data = try Data(contentsOf: dataFile)
                    data[0] ^= 1
                    try data.write(to: dataFile)
                case 1: try Data([0]).write(to: dataFile)
                case 2: try Data("{broken".utf8).write(to: manifestFile)
                case 3: try Data(repeating: 0, count: 4097).write(to: manifestFile)
                default: try editManifest(directory, resign: false) { $0["tokens"] = [7, 8] }
                }
                XCTAssertNil(
                    try disk.restore(
                        prompt: [1, 2, 3], maximumPrefixTokens: 2, prototype: [KVCacheSimple()]))
                XCTAssertEqual(disk.entryCount, 0)
            }
        }
    }

    func testResignedInvalidManifestFieldsAreRejectedBeforeArrayConstruction() throws {
        for mutation in 0 ..< 10 {
            try withDirectory { directory in
                let disk = try store(directory)
                try disk.store(tokens: [1, 2], cache: [cache([1, 2])])
                try editManifest(directory) { manifest in
                    switch mutation {
                    case 0: manifest["version"] = 999
                    case 1: manifest["layout"] = "different"
                    case 2: manifest["tokens"] = [8, 9]
                    case 3: manifest["dataFile"] = "../../outside.bin"
                    case 4: manifest["dataBytes"] = Int.max
                    default:
                        var layers = try XCTUnwrap(manifest["layers"] as? [[String: Any]])
                        if mutation == 5 { layers[0]["offset"] = 1 }
                        if mutation == 6 { layers[0]["kind"] = "arbitrary-class" }
                        if mutation >= 7 {
                            var tensors = try XCTUnwrap(layers[0]["tensors"] as? [[String: Any]])
                            if mutation == 7 { tensors[0]["shape"] = [1, 1, Int.max, 4] }
                            if mutation == 8 { tensors[0]["dtype"] = "invalid-dtype" }
                            if mutation == 9 { tensors[0]["byteCount"] = -1 }
                            layers[0]["tensors"] = tensors
                        }
                        manifest["layers"] = layers
                    }
                }
                XCTAssertNil(
                    try disk.restore(
                        prompt: [1, 2, 3], maximumPrefixTokens: 2, prototype: [KVCacheSimple()]))
            }
        }
    }

    func testInvalidRecurrentSlotsAndOffsetsAreRejected() throws {
        for mutation in 0 ..< 3 {
            try withDirectory { directory in
                let recurrent = MambaCache()
                recurrent[1] = MLXArray.ones([1, 2, 3])
                recurrent.offset = 2
                let disk = try store(directory, layout: "mamba")
                try disk.store(tokens: [1, 2], cache: [recurrent])
                try editManifest(directory) { manifest in
                    var layers = try XCTUnwrap(manifest["layers"] as? [[String: Any]])
                    if mutation == 0 { layers[0]["offset"] = -1 }
                    if mutation == 1 { layers[0]["metadata"] = ["2", "-1"] }
                    if mutation == 2 { layers[0]["metadata"] = ["999999999", "1"] }
                    manifest["layers"] = layers
                }
                XCTAssertNil(
                    try disk.restore(
                        prompt: [1, 2, 3], maximumPrefixTokens: 2, prototype: [MambaCache()]))
            }
        }
    }

    func testPrototypeTopologyShapeDtypeAndQuantizationMustMatch() throws {
        try withDirectory { directory in
            let disk = try store(directory)
            let prototypes: [[KVCache]] = [
                [MambaCache()], [KVCacheSimple(), KVCacheSimple()], [cache([1], width: 8)],
            ]
            for prototype in prototypes {
                try disk.store(tokens: [1, 2], cache: [cache([1, 2])])
                XCTAssertNil(
                    try disk.restore(
                        prompt: [1, 2, 3], maximumPrefixTokens: 2, prototype: prototype))
            }
            let differentDtype = cache([1])
            differentDtype.state = differentDtype.state.map { $0.asType(.float16) }
            try disk.store(tokens: [1, 2], cache: [cache([1, 2])])
            XCTAssertNil(
                try disk.restore(
                    prompt: [1, 2, 3], maximumPrefixTokens: 2, prototype: [differentDtype]))
            let quantized = try cache([1, 2], width: 64).toQuantized(groupSize: 32, bits: 4)
            try disk.store(tokens: [1, 2], cache: [quantized])
            XCTAssertNil(
                try disk.restore(
                    prompt: [1, 2, 3], maximumPrefixTokens: 2,
                    prototype: [QuantizedKVCache(groupSize: 64, bits: 8)]))
        }
    }

    func testDiskBudgetEvictsLeastRecentlyUsedAcrossRestart() throws {
        try withDirectory { directory in
            let measuring = try store(directory)
            try measuring.store(tokens: [1, 2], cache: [cache([1, 2])])
            let entryBytes = measuring.storedBytes
            try measuring.clear()
            let limit = entryBytes * 2 + 32
            do {
                let writer = try store(directory, maximumBytes: limit)
                try writer.store(tokens: [1, 2], cache: [cache([1, 2])])
                try writer.store(tokens: [3, 4], cache: [cache([3, 4])])
                XCTAssertNotNil(
                    try writer.restore(
                        prompt: [1, 2, 9], maximumPrefixTokens: 2, prototype: [KVCacheSimple()]))
            }
            let reader = try store(directory, maximumBytes: limit)
            try reader.store(tokens: [5, 6], cache: [cache([5, 6])])
            XCTAssertLessThanOrEqual(reader.storedBytes, limit)
            XCTAssertEqual(reader.entryCount, 2)
            XCTAssertNotNil(
                try reader.restore(
                    prompt: [1, 2, 9], maximumPrefixTokens: 2, prototype: [KVCacheSimple()]))
            XCTAssertNil(
                try reader.restore(
                    prompt: [3, 4, 9], maximumPrefixTokens: 2, prototype: [KVCacheSimple()]))
            XCTAssertNotNil(
                try reader.restore(
                    prompt: [5, 6, 9], maximumPrefixTokens: 2, prototype: [KVCacheSimple()]))
            let namespace = try XCTUnwrap(files(directory, prefix: "entry-v1-").first)
                .deletingLastPathComponent()
            let owned = try FileManager.default.contentsOfDirectory(
                at: namespace, includingPropertiesForKeys: nil
            )
            .filter { $0.lastPathComponent != "owner-v1.json" }
            let actualBytes = try owned.reduce(0) { try $0 + Data(contentsOf: $1).count }
            XCTAssertLessThanOrEqual(actualBytes, limit)
        }
    }

    func testRecoveryCleansInterruptedWritesAndPreservesUnownedFiles() throws {
        try withDirectory { directory in
            let disk = try store(directory)
            try disk.store(tokens: [1, 2], cache: [cache([1, 2])])
            let namespace = try XCTUnwrap(files(directory, prefix: "entry-v1-").first)
                .deletingLastPathComponent()
            let orphan = namespace.appendingPathComponent(
                "data-v1-\(UUID().uuidString.lowercased()).bin")
            let temporary = namespace.appendingPathComponent(
                "temp-v1-\(UUID().uuidString.lowercased()).tmp")
            let unrelated = namespace.appendingPathComponent("keep-me.txt")
            for file in [orphan, temporary, unrelated] { try Data([1, 2, 3]).write(to: file) }
            let restarted = try store(directory)
            XCTAssertEqual(restarted.entryCount, 1)
            XCTAssertFalse(FileManager.default.fileExists(atPath: orphan.path))
            XCTAssertFalse(FileManager.default.fileExists(atPath: temporary.path))
            try restarted.clear()
            XCTAssertEqual(restarted.entryCount, 0)
            XCTAssertEqual(try Data(contentsOf: unrelated), Data([1, 2, 3]))
        }
    }

    func testSymlinksCannotRedirectReadsOrCleanup() throws {
        try withDirectory { directory in
            let root = directory.appendingPathComponent("root", isDirectory: true)
            let disk = try store(root)
            try disk.store(tokens: [1, 2], cache: [cache([1, 2])])
            let dataFile = try XCTUnwrap(files(root, prefix: "data-v1-").first)
            let outside = directory.appendingPathComponent("outside.bin")
            let original = try Data(contentsOf: dataFile)
            try original.write(to: outside)
            try FileManager.default.removeItem(at: dataFile)
            try FileManager.default.createSymbolicLink(at: dataFile, withDestinationURL: outside)
            XCTAssertNil(
                try disk.restore(
                    prompt: [1, 2, 3], maximumPrefixTokens: 2, prototype: [KVCacheSimple()]))
            XCTAssertEqual(try Data(contentsOf: outside), original)
            let link = directory.appendingPathComponent("linked-root")
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: root)
            XCTAssertThrowsError(try store(link))
            XCTAssertThrowsError(try store(link.appendingPathComponent("child")))
            XCTAssertFalse(
                FileManager.default.fileExists(atPath: root.appendingPathComponent("child").path))
        }
    }

    func testUnsupportedCacheAndOversizeSnapshotThrowWithoutPublishing() throws {
        final class Unsupported: KVCacheSimple {}
        try withDirectory { directory in
            let disk = try store(directory, maximumBytes: 2048)
            XCTAssertThrowsError(try disk.store(tokens: [1], cache: [Unsupported()]))
            XCTAssertThrowsError(try disk.store(tokens: [1], cache: [MambaCache()]))
            XCTAssertThrowsError(try disk.store(tokens: [1], cache: [cache([1, 2])]))
            XCTAssertThrowsError(
                try disk.store(
                    tokens: Array(1 ... 128), cache: [cache(Array(1 ... 128), width: 64)]))
            XCTAssertEqual(disk.entryCount, 0)
            XCTAssertTrue(try files(directory, prefix: "entry-v1-").isEmpty)
            XCTAssertTrue(try files(directory, prefix: "data-v1-").isEmpty)
        }
    }
}
