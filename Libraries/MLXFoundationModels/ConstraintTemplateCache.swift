// Copyright © 2026 Faux Fox.

#if FoundationModelsIntegration
#if canImport(FoundationModels, _version: 2)

/// Bounds retained template keys while callers use independent matchers.
struct ConstraintTemplateCache<Value> {
    let maximumEntries: Int
    let maximumSourceBytes: Int

    private var entries: [String: (value: Value, sourceBytes: Int)] = [:]
    private var oldestFirst: [String] = []
    private(set) var retainedSourceBytes = 0

    var count: Int { entries.count }

    init(maximumEntries: Int, maximumSourceBytes: Int) {
        precondition(maximumEntries > 0 && maximumSourceBytes > 0)
        self.maximumEntries = maximumEntries
        self.maximumSourceBytes = maximumSourceBytes
    }

    func canRetain(_ key: String) -> Bool {
        key.utf8.count <= maximumSourceBytes
    }

    mutating func value(for key: String) -> Value? {
        guard let entry = entries[key] else { return nil }
        oldestFirst.removeAll { $0 == key }
        oldestFirst.append(key)
        return entry.value
    }

    @discardableResult
    mutating func insert(_ value: Value, for key: String) -> Bool {
        let bytes = key.utf8.count
        guard bytes <= maximumSourceBytes else { return false }
        removeValue(for: key)
        entries[key] = (value, bytes)
        oldestFirst.append(key)
        retainedSourceBytes += bytes
        while count > maximumEntries || retainedSourceBytes > maximumSourceBytes {
            removeValue(for: oldestFirst[0])
        }
        return true
    }

    @discardableResult
    mutating func removeValue(for key: String) -> Value? {
        guard let removed = entries.removeValue(forKey: key) else { return nil }
        oldestFirst.removeAll { $0 == key }
        retainedSourceBytes -= removed.sourceBytes
        return removed.value
    }

    mutating func removeAll(where predicate: (String) -> Bool) {
        for key in oldestFirst.filter(predicate) { removeValue(for: key) }
    }

    mutating func removeAll() {
        entries.removeAll()
        oldestFirst.removeAll()
        retainedSourceBytes = 0
    }
}

#endif
#endif
