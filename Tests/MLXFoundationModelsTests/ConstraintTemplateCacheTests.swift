// Copyright © 2026 Faux Fox.

import Testing

@testable import MLXFoundationModels

#if FoundationModelsIntegration && canImport(FoundationModels, _version: 2)

@Suite("Constraint template cache")
struct ConstraintTemplateCacheTests {
    @Test("evicts the least recently used entry at the count limit")
    func countLimit() {
        var cache = ConstraintTemplateCache<Int>(maximumEntries: 2, maximumSourceBytes: 100)
        cache.insert(1, for: "a")
        cache.insert(2, for: "b")
        let initial = cache.value(for: "a")
        #expect(initial == 1)

        cache.insert(3, for: "c")

        let evicted = cache.value(for: "b")
        let recent = cache.value(for: "a")
        let newest = cache.value(for: "c")
        #expect(cache.count == 2)
        #expect(evicted == nil)
        #expect(recent == 1)
        #expect(newest == 3)
    }

    @Test("counts UTF-8 bytes and does not retain oversized keys")
    func byteLimit() {
        var cache = ConstraintTemplateCache<Int>(maximumEntries: 10, maximumSourceBytes: 5)
        cache.insert(1, for: "é")
        cache.insert(2, for: "abc")
        #expect(cache.retainedSourceBytes == 5)

        cache.insert(3, for: "d")
        let evicted = cache.value(for: "é")
        #expect(evicted == nil)
        #expect(cache.retainedSourceBytes == 4)

        #expect(!cache.canRetain("123456"))
        let inserted = cache.insert(4, for: "123456")
        #expect(!inserted)
        #expect(cache.count == 2)
        #expect(cache.retainedSourceBytes == 4)
    }

    @Test("replacing and removing entries keeps byte accounting correct")
    func replacementAndRemoval() {
        var cache = ConstraintTemplateCache<Int>(maximumEntries: 3, maximumSourceBytes: 30)
        cache.insert(1, for: "model-a:one")
        cache.insert(2, for: "model-b:two")
        cache.insert(3, for: "model-a:one")
        #expect(cache.count == 2)
        #expect(cache.retainedSourceBytes == 22)

        cache.removeAll { $0.hasPrefix("model-a:") }
        let removed = cache.value(for: "model-a:one")
        let survivor = cache.value(for: "model-b:two")
        #expect(removed == nil)
        #expect(survivor == 2)

        cache.removeAll()
        #expect(cache.count == 0)
        #expect(cache.retainedSourceBytes == 0)
    }
}

#endif
