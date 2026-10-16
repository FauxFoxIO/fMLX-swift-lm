// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

// MARK: - Concurrent weight loading

// MLX's safetensors loader is lazy: `loadArraysAndMetadata` reads only the header, and a
// tensor's bytes are read when its array is evaluated. A single `eval` of everything
// serializes that I/O and the copies into unified memory no matter how fast the disk is.
// Splitting each file into contiguous byte-balanced ranges and evaluating the ranges from
// concurrent `eval` calls overlaps read, copy, and allocation. Measured on an M4 Pro
// (14 cores, 5 GB shards, NVMe at ~6.1 GB/s sequential): the serial loader moves ~3-4.5 GB/s
// while the concurrent one reaches the disk ceiling cold (~5.9 GB/s) and >10 GB/s from the
// page cache -- a 30-45% faster cold load, about 2x warm. `F_RDADVISE`/read-ahead variants
// measured *slower* than the serial baseline because the advised I/O competes with the
// loader's own reads.

/// One tensor's byte range in a safetensors file, from the file's own header.
struct SafetensorSpan {
    let name: String
    let byteCount: Int64
}

/// The tensors of the safetensors file at `url`, ordered by their position in the file.
///
/// Reads the 8-byte header length and the JSON header only. Throws when the file is not a
/// well-formed safetensors file; callers fall back to loading the file whole.
func safetensorSpansInFileOrder(url: URL) throws -> [SafetensorSpan] {
    struct Malformed: Error {}

    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }

    guard let lengthData = try handle.read(upToCount: 8), lengthData.count == 8 else {
        throw Malformed()
    }
    let headerLength = lengthData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        .littleEndian
    // a header bigger than this is not a header
    guard headerLength > 0, headerLength <= 512 * 1024 * 1024 else { throw Malformed() }
    guard let headerData = try handle.read(upToCount: Int(headerLength)),
        headerData.count == headerLength,
        let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
    else {
        throw Malformed()
    }

    var spans = [(name: String, begin: Int64, byteCount: Int64)]()
    for (name, value) in header {
        guard name != "__metadata__" else { continue }
        guard let entry = value as? [String: Any],
            let offsets = entry["data_offsets"] as? [Any], offsets.count == 2,
            let begin = (offsets[0] as? NSNumber)?.int64Value,
            let end = (offsets[1] as? NSNumber)?.int64Value,
            end >= begin
        else {
            throw Malformed()
        }
        spans.append((name, begin, end - begin))
    }
    spans.sort { $0.begin < $1.begin }
    return spans.map { SafetensorSpan(name: $0.name, byteCount: $0.byteCount) }
}

/// Contiguous index ranges of `byteCounts` whose byte totals are balanced around
/// `total / groupCount`, preserving order.
func contiguousLoadGroups(byteCounts: [Int64], groupCount: Int) -> [Range<Int>] {
    guard !byteCounts.isEmpty else { return [] }
    let total = byteCounts.reduce(0, +)
    guard groupCount > 1, total > 0 else { return [0 ..< byteCounts.count] }

    let groups = Int64(groupCount)
    var ranges = [Range<Int>]()
    var start = 0
    var cumulative: Int64 = 0
    var boundary: Int64 = 1
    for (index, byteCount) in byteCounts.enumerated() {
        cumulative += byteCount
        if boundary < groups, cumulative >= total * boundary / groups {
            ranges.append(start ..< index + 1)
            start = index + 1
            boundary += 1
        }
    }
    if start < byteCounts.count {
        ranges.append(start ..< byteCounts.count)
    }
    return ranges
}

/// How many concurrent evaluations to spread a model's weight loading across.
///
/// Throughput rises with concurrent readers until the disk (cold) or the memory system (warm)
/// saturates -- around 8-16 in-flight readers on Apple silicon. More workers than cores only
/// adds contention.
func weightLoadConcurrency(processorCount: Int = ProcessInfo.processInfo.activeProcessorCount)
    -> Int
{
    max(4, min(16, processorCount))
}

/// Below this size a file is loaded whole: splitting cannot beat a single sequential read.
private let minimumBytesPerLoadGroup: Int64 = 256 * 1024 * 1024

/// Selects checkpoint tensors before their lazy safetensors arrays are evaluated.
///
/// Most callers use ``all``.  Checkpoint components that share files can select their own
/// namespace, avoiding materializing the rest of a combined checkpoint only to discard it in
/// `sanitize(weights:)`.
package struct WeightTensorNameSelection: @unchecked Sendable {
    private let matches: (String) -> Bool
    fileprivate let filtersNames: Bool

    package static let all = Self(matches: { _ in true }, filtersNames: false)

    private init(
        matches: @escaping (String) -> Bool, filtersNames: Bool
    ) {
        self.matches = matches
        self.filtersNames = filtersNames
    }

    package static func prefixed(_ prefix: String) -> Self {
        Self(matches: { $0.hasPrefix(prefix) }, filtersNames: true)
    }

    /// Select every tensor except names accepted by `predicate`.
    ///
    /// This is used by streamed checkpoint components to keep their lazy
    /// safetensor arrays out of the normal materialization path.
    package static func excluding(
        _ predicate: @escaping (String) -> Bool
    ) -> Self {
        Self(matches: { !predicate($0) }, filtersNames: true)
    }

    package func intersecting(
        _ predicate: @escaping (String) -> Bool
    ) -> Self {
        Self(matches: { self.matches($0) && predicate($0) }, filtersNames: true)
    }

    package func contains(_ name: String) -> Bool {
        matches(name)
    }
}

/// Lock-guarded shared state for the concurrent load.
private final class ConcurrentLoadState: @unchecked Sendable {
    private let lock = NSLock()
    private var perFile: [[String: MLXArray]]
    private var perFileMetadata: [[String: String]]
    private var firstError: Error?

    init(fileCount: Int) {
        perFile = Array(repeating: [:], count: fileCount)
        perFileMetadata = Array(repeating: [:], count: fileCount)
    }

    func merge(file: Int, weights: [String: MLXArray], metadata: [String: String]) {
        lock.lock()
        defer { lock.unlock() }
        perFile[file].merge(weights) { _, new in new }
        perFileMetadata[file] = metadata
    }

    func record(error: Error) {
        lock.lock()
        defer { lock.unlock() }
        if firstError == nil { firstError = error }
    }

    /// Preserve file order and tensor metadata, including later-file-wins duplicates.
    func result() throws -> ModelCheckpoint {
        lock.lock()
        defer { lock.unlock() }
        if let firstError { throw firstError }
        var weights = [String: MLXArray]()
        var weightMetadata = [String: [String: String]]()
        for (index, fileWeights) in perFile.enumerated() {
            weights.merge(fileWeights) { _, new in new }
            for name in fileWeights.keys {
                weightMetadata[name] = perFileMetadata[index]
            }
        }
        let metadata = perFileMetadata.first { !$0.isEmpty } ?? [:]
        return ModelCheckpoint(weights: weights, metadata: metadata, weightMetadata: weightMetadata)
    }
}

/// Load and materialize the weights of every file in `urls`, evaluating contiguous byte
/// ranges of each file concurrently.
///
/// Each work item lazily opens its file, evaluates only its assigned tensors (forcing that
/// range's I/O inside the work item), and the results are merged in file order. A file whose
/// header cannot be parsed is loaded whole by one work item, which is exactly the serial
/// loader's behavior for that file.
func loadWeightArrays(urls: [URL]) throws -> (
    weights: [String: MLXArray], metadata: [String: String]
) {
    let checkpoint = try loadModelCheckpoint(urls: urls)
    return (checkpoint.weights, checkpoint.metadata)
}

/// Tensors under `excludedNamespaces` are never read. A file that holds only excluded tensors
/// still supplies its metadata.
func loadModelCheckpoint(urls: [URL], excludedNamespaces: [String] = []) throws
    -> ModelCheckpoint
{
    let exclusions = CheckpointNameMapping(excludedNamespaces.map { .excludePrefix($0) })
    return try loadModelCheckpoint(urls: urls) { exclusions.mapName($0) != nil }
}

/// Only tensors whose names pass `isRead` are read. With `materialize` false the arrays stay
/// lazy, so a tensor-parallel rank can slice them before any bytes are read.
func loadModelCheckpoint(
    urls: [URL], materialize: Bool = true, isRead: @escaping @Sendable (String) -> Bool
) throws
    -> ModelCheckpoint
{
    struct WorkItem {
        let file: Int
        let url: URL
        /// tensors this item evaluates; nil evaluates every tensor in the file that is read
        let names: [String]?
    }

    let items: [WorkItem] = {
        var spansPerFile = [[SafetensorSpan]?]()
        var totalBytes: Int64 = 0
        for url in urls {
            let spans = (try? safetensorSpansInFileOrder(url: url))?.filter { isRead($0.name) }
            spansPerFile.append(spans)
            totalBytes += spans?.reduce(0) { $0 + $1.byteCount } ?? 0
        }

        let concurrency = weightLoadConcurrency()
        let groupBytes = max(minimumBytesPerLoadGroup, totalBytes / Int64(concurrency))
        var items = [WorkItem]()
        for (file, url) in urls.enumerated() {
            if let spans = spansPerFile[file], !spans.isEmpty {
                let bytes = spans.reduce(0) { $0 + $1.byteCount }
                let groupCount = max(1, Int(bytes / groupBytes))
                for range in contiguousLoadGroups(
                    byteCounts: spans.map(\.byteCount), groupCount: groupCount)
                {
                    items.append(
                        WorkItem(file: file, url: url, names: spans[range].map(\.name)))
                }
            } else {
                // An empty list reads only the metadata of a file with nothing to read.
                let names: [String]? = spansPerFile[file] == nil ? nil : []
                items.append(WorkItem(file: file, url: url, names: names))
            }
        }
        return items
    }()

    let state = ConcurrentLoadState(fileCount: urls.count)
    DispatchQueue.concurrentPerform(iterations: items.count) { index in
        let item = items[index]
        do {
            // A caller's withError does not reach this worker thread.
            try withLoaderThreadError {
                // Explicitly the CPU stream: `Load` has no GPU implementation and the arrays
                // land in unified memory either way. The concurrency comes from evaluating
                // disjoint groups from many threads, not from the stream itself.
                let (all, metadata) = try loadArraysAndMetadata(url: item.url, stream: .cpu)

                var selected = [String: MLXArray]()
                if let names = item.names {
                    for name in names {
                        if let array = all[name] { selected[name] = array }
                    }
                } else {
                    selected = all.filter { isRead($0.key) }
                }

                // force this range's I/O here, on this stream, in file-offset order
                if materialize, !selected.isEmpty { eval(Array(selected.values)) }
                state.merge(file: item.file, weights: selected, metadata: metadata)
            }
        } catch {
            state.record(error: error)
        }
    }
    return try state.result()
}

private struct SafetensorsIndex: Decodable {
    let weightMap: [String: String]

    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}

private struct IndexedWeightFiles {
    let urls: [URL]
    let weightMap: [String: String]
}

/// How the safetensors files holding a model's weights are chosen.
///
/// ## See Also
/// - ``ModelConfiguration/weightFileSelection``
public enum WeightFileSelection: Sendable, Equatable {
    /// Use `model.safetensors.index.json` when it names files that exist, otherwise the
    /// conventional `model*.safetensors` (then `weight*.safetensors`) names.
    ///
    /// This is what a well-packaged checkpoint wants and it is the default.
    case automatic

    /// Load every safetensors file in the model directory.
    ///
    /// This is an escape hatch for a checkpoint whose index is known to be wrong in a way
    /// ``automatic`` cannot detect -- an index that names files that all exist but omits
    /// weights the model needs. It loads files that may not belong to this model, and a
    /// stray tensor whose name collides with one the model's `sanitize(weights:)` rewrites
    /// is loaded silently rather than reported, so prefer a model that declares its own
    /// extra files (see ``AdditionalWeightFilesProviding``) where that is possible.
    case allFilesPresent
}

/// The safetensors files in `modelDirectory` that hold the model's weights.
///
/// Only the top level of the directory is considered. Checkpoints keep auxiliary weights that
/// belong to a different module in subdirectories (for example `mlx-community/Qwen3.5-4B-OptiQ-4bit`
/// and its `optiq/mtp.safetensors`), and a nested Hugging Face snapshot cache under a local
/// checkpoint directory would otherwise be pulled in as well.
///
/// With ``WeightFileSelection/automatic`` the files are chosen in this order:
///
/// 1. The files named by `model.safetensors.index.json`, when it exists and every file it names
///    exists. The index is precise about which of several weight files belong to the model, which
///    matters for a repo that ships both a consolidated file and shards.
/// 2. The conventional `model*.safetensors` names, matching `mlx_lm.utils.load_model`. Uploads
///    regularly ship an index carried over from an unquantized source repo that names shards the
///    repo does not contain, and the convention is what those repos actually follow.
/// 3. `weight*.safetensors`, then every safetensors file present, so a directory that follows no
///    convention at all still loads.
///
/// `additionalFiles` names files the model requires that no rule above selects, for example the
/// Jina reranker's `projector.safetensors`. They are appended, so a file the index already names
/// is not loaded twice, and names that are not present are ignored.
///
/// - Parameters:
///   - modelDirectory: directory holding the weight files
///   - selection: how to choose the files, see ``WeightFileSelection``
///   - additionalFiles: file names, relative to `modelDirectory`, to load in addition to the
///     selected ones. See ``AdditionalWeightFilesProviding/additionalWeightFiles``.
package func safetensorWeightURLs(
    in modelDirectory: URL,
    selection: WeightFileSelection = .automatic,
    additionalFiles: [String] = [],
    tensorNameSelection: WeightTensorNameSelection = .all
) throws -> [URL] {
    let present = topLevelSafetensorURLs(in: modelDirectory)

    let selected: [URL]
    let indexed: IndexedWeightFiles?
    switch selection {
    case .allFilesPresent:
        selected = present
        indexed = nil
    case .automatic:
        indexed = try indexedWeightFiles(in: modelDirectory)
        selected = indexed?.urls ?? conventionalWeightURLs(in: present)
    }

    var seen = Set(selected.map(\.standardizedFileURL.path))
    var urls = selected
    for name in additionalFiles {
        let url = modelDirectory.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path),
            seen.insert(url.standardizedFileURL.path).inserted
        else {
            continue
        }
        urls.append(url)
    }

    // A valid index identifies which shards can contain a selected tensor, so avoid even
    // opening unrelated shards.  Additional files remain eligible because an index does not
    // describe them.  Headers still perform the authoritative per-tensor filtering below.
    if tensorNameSelection.filtersNames, let indexed {
        let indexedPaths = Set(indexed.urls.map(\.standardizedFileURL.path))
        let selectedPaths = Set(
            indexed.weightMap.compactMap { name, file in
                tensorNameSelection.contains(name)
                    ? modelDirectory.appendingPathComponent(file).standardizedFileURL.path : nil
            })
        urls.removeAll { url in
            let path = url.standardizedFileURL.path
            return indexedPaths.contains(path) && !selectedPaths.contains(path)
        }
    }
    return urls
}

/// The files named by `model.safetensors.index.json`, or `nil` when there is no index or it
/// names a file the directory does not contain.
///
/// Existence is checked against the file system rather than the top-level listing: an index may
/// legitimately map weights into a subdirectory, and that is a deliberate statement about where
/// this model's weights live rather than an unrelated file that happens to be nearby.
private func indexedWeightFiles(in modelDirectory: URL) throws -> IndexedWeightFiles? {
    let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
    guard FileManager.default.fileExists(atPath: indexURL.path) else {
        return nil
    }

    let data = try Data(contentsOf: indexURL)
    let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
    let urls = Set(index.weightMap.values)
        .sorted()
        .map { modelDirectory.appendingPathComponent($0) }

    guard !urls.isEmpty,
        urls.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) })
    else {
        return nil
    }
    return IndexedWeightFiles(urls: urls, weightMap: index.weightMap)
}

/// The conventionally named weight files among `present`, matching `mlx_lm.utils.load_model`'s
/// `model*.safetensors` glob, with `weight*.safetensors` and then everything as fallbacks.
private func conventionalWeightURLs(in present: [URL]) -> [URL] {
    for prefix in ["model", "weight"] {
        let matches = present.filter { $0.lastPathComponent.hasPrefix(prefix) }
        if !matches.isEmpty {
            return matches
        }
    }
    return present
}

private func topLevelSafetensorURLs(in modelDirectory: URL) -> [URL] {
    let contents =
        (try? FileManager.default.contentsOfDirectory(
            at: modelDirectory, includingPropertiesForKeys: nil)) ?? []
    return
        contents
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }
}

/// Reports whether selected checkpoint files contain a tensor accepted by `tensorNameSelection`.
///
/// This reads safetensor headers only.  It is useful for optional checkpoint components, whose
/// presence must be established without materializing any of the target model's tensors.
package func containsWeightTensor(
    in modelDirectory: URL,
    selection: WeightFileSelection = .automatic,
    additionalFiles: [String] = [],
    tensorNameSelection: WeightTensorNameSelection
) throws -> Bool {
    let urls = try safetensorWeightURLs(
        in: modelDirectory,
        selection: selection,
        additionalFiles: additionalFiles,
        tensorNameSelection: tensorNameSelection)
    return try urls.contains { url in
        try safetensorSpansInFileOrder(url: url).contains {
            tensorNameSelection.contains($0.name)
        }
    }
}

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads model weight `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/prepareCheckpoint(_:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights. Derived inference-only state is prepared after the
/// checkpoint update and before the model is evaluated and returned to callers.
///
/// The weight files are chosen from `model.safetensors.index.json` when it names files that
/// exist, and otherwise by the conventional `model*.safetensors` names. A model can name extra
/// files it needs by conforming to ``AdditionalWeightFilesProviding``, and a caller can override
/// the choice with ``ModelConfiguration/weightFileSelection``.
///
/// The model loads without the `excludedComponents` it supports, see
/// ``ModelComponentsProviding``. Its on-demand components load the first time an input
/// needs them, unless `componentLoading` is ``ComponentLoading/immediate``. Tensors of components
/// that do not load now and of the model's ``BaseLanguageModel/excludedCheckpointNamespaces`` are
/// not read.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    weightFileSelection: WeightFileSelection = .automatic,
    excludedComponents: Set<ModelComponent> = [],
    componentLoading: ComponentLoading = .onFirstUse
) throws {
    try loadWeights(
        modelDirectory: modelDirectory, model: model, tensorNameSelection: .all,
        quantization: quantization, perLayerQuantization: perLayerQuantization,
        weightFileSelection: weightFileSelection, excludedComponents: excludedComponents,
        componentLoading: componentLoading)
}

/// Package-internal variant for a checkpoint component embedded beside another model.
///
/// The name selection happens before lazy safetensor arrays are evaluated.  The normal public
/// entry point always selects every weight.
package func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    tensorNameSelection: WeightTensorNameSelection,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    weightFileSelection: WeightFileSelection = .automatic,
    excludedComponents: Set<ModelComponent> = [],
    componentLoading: ComponentLoading = .onFirstUse,
    tensorParallelGroup: MLXDistributed.Group? = nil
) throws {
    // a group of one runs the unsharded model
    let shardingGroup = tensorParallelGroup.flatMap { $0.size > 1 ? $0 : nil }
    let shardable = model as? any ShardableModel
    if shardingGroup != nil, shardable == nil {
        throw TensorParallelError.modelNotShardable(String(describing: type(of: model)))
    }

    if let preparing = model as? any ModelArtifactPreparing {
        try preparing.prepareArtifact(in: modelDirectory)
    }

    let effectiveTensorNameSelection: WeightTensorNameSelection
    if let selecting = model as? any WeightTensorSelecting {
        effectiveTensorNameSelection = tensorNameSelection.intersecting {
            selecting.shouldLoadWeightTensor(named: $0)
        }
    } else {
        effectiveTensorNameSelection = tensorNameSelection
    }

    let additionalFiles = (model as? any AdditionalWeightFilesProviding)?.additionalWeightFiles
    let weightURLs = try safetensorWeightURLs(
        in: modelDirectory,
        selection: weightFileSelection,
        additionalFiles: additionalFiles ?? [],
        tensorNameSelection: effectiveTensorNameSelection)
    let perLayerQuantization =
        perLayerQuantization
        ?? quantization.map { .init(quantization: $0, perLayerQuantization: [:]) }
    let excludedNamespaces =
        try prepareComponents(
            of: model, urls: weightURLs, excluded: excludedComponents,
            loading: componentLoading, perLayerQuantization: perLayerQuantization)
        + model.excludedCheckpointNamespaces
    let exclusions = CheckpointNameMapping(excludedNamespaces.map { .excludePrefix($0) })
    var checkpoint = try loadModelCheckpoint(
        urls: weightURLs, materialize: shardingGroup == nil
    ) {
        [effectiveTensorNameSelection] name in
        exclusions.mapName(name) != nil && effectiveTensorNameSelection.contains(name)
    }
    checkpoint.perLayerQuantization = perLayerQuantization
    checkpoint = try model.prepareCheckpoint(checkpoint)
    try applyCheckpoint(checkpoint, to: model)
    // Derived fused projections replace their source modules. Drop loader-owned
    // references first so each replaced source can be reclaimed immediately.
    checkpoint.weights.removeAll(keepingCapacity: false)

    if let shardingGroup, let shardable {
        // Slice the lazy weights, then read them in batches. Each full tensor is freed once
        // its slice is copied, so a rank never holds the whole model.
        try shardable.shard(group: shardingGroup)
        evaluateInBatches(model.parameters().flattened().map(\.1))
    }

    if let preparing = model as? any LoadedWeightsPreparing {
        try preparing.prepareLoadedWeights()
    }

    // Build derived inference-only state and realize the model while the loader
    // still has exclusive access. Forward passes must remain read-only.
    materializeModelForInference(model)
    // Fused projections replace materialized checkpoint buffers with larger
    // concatenations. Return the retired buffers instead of retaining them in MLX's cache.
    MLX.Memory.clearCache()

    if let shardingGroup {
        // ranks finish loading at different times; meet before the first collective forward
        MLXDistributed.barrier(group: shardingGroup)
    }
}

/// Quantize the layers `checkpoint` has scales for, then update `module` with strict validation.
func applyCheckpoint(_ checkpoint: ModelCheckpoint, to module: Module) throws {
    let weights = checkpoint.weights
    if let perLayerQuantization = checkpoint.perLayerQuantization {
        quantize(model: module) { path, _ in
            guard weights["\(path).scales"] != nil else { return nil }
            return perLayerQuantization.quantization(layer: path)?.asTuple
        }
    }
    try module.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
}

/// Async variant of
/// ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:weightFileSelection:excludedComponents:componentLoading:)-858j8``.
///
/// Loading blocks its thread on file I/O and fans out with `DispatchQueue.concurrentPerform`.
/// Swift concurrency's cooperative threads must never block, so this overload runs the load
/// on a global queue and suspends the caller instead. Async callers resolve to this overload;
/// the synchronous one remains for synchronous code such as model conversion.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    weightFileSelection: WeightFileSelection = .automatic,
    excludedComponents: Set<ModelComponent> = [],
    componentLoading: ComponentLoading = .onFirstUse
) async throws {
    try await loadWeights(
        modelDirectory: modelDirectory, model: model, tensorNameSelection: .all,
        quantization: quantization, perLayerQuantization: perLayerQuantization,
        weightFileSelection: weightFileSelection, excludedComponents: excludedComponents,
        componentLoading: componentLoading)
}

/// Load model weights and shard them across a tensor-parallel group.
///
/// Every rank in `tensorParallelGroup` calls this with the same checkpoint. The checkpoint is
/// loaded lazily, the model's ``ShardableModel/shard(group:)`` slices it, and only this
/// rank's slice is read into memory. A group of size one loads the whole model.
///
/// Throws ``TensorParallelError/modelNotShardable(_:)`` when the group has more than one rank
/// and the model does not conform to ``ShardableModel``.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    tensorParallelGroup: MLXDistributed.Group,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    weightFileSelection: WeightFileSelection = .automatic,
    excludedComponents: Set<ModelComponent> = []
) async throws {
    try await loadWeights(
        modelDirectory: modelDirectory, model: model, tensorNameSelection: .all,
        perLayerQuantization: perLayerQuantization,
        weightFileSelection: weightFileSelection, excludedComponents: excludedComponents,
        tensorParallelGroup: tensorParallelGroup)
}

/// Async package-internal variant of the selective weight loader.
package func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    tensorNameSelection: WeightTensorNameSelection,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil,
    weightFileSelection: WeightFileSelection = .automatic,
    excludedComponents: Set<ModelComponent> = [],
    componentLoading: ComponentLoading = .onFirstUse,
    tensorParallelGroup: MLXDistributed.Group? = nil
) async throws {
    let model = SendableBox(model)
    try await withCheckedThrowingContinuation {
        (continuation: CheckedContinuation<Void, any Error>) in
        DispatchQueue.global(qos: .userInitiated).async {
            // The caller's withError does not reach this thread.
            continuation.resume(
                with: Result {
                    try withLoaderThreadError {
                        try loadWeights(
                            modelDirectory: modelDirectory, model: model.consume(),
                            tensorNameSelection: tensorNameSelection,
                            quantization: quantization,
                            perLayerQuantization: perLayerQuantization,
                            weightFileSelection: weightFileSelection,
                            excludedComponents: excludedComponents,
                            componentLoading: componentLoading,
                            tensorParallelGroup: tensorParallelGroup)
                    }
                })
        }
    }
}

/// Run `body` in an MLX error scope on the current thread. An MLX error is thrown
/// in place of a later error it caused in `body`, such as a shape mismatch.
func withLoaderThreadError<R>(_ body: () throws -> R) throws -> R {
    try withError { mlxError in
        do {
            return try body()
        } catch {
            try mlxError.check()
            throw error
        }
    }
}
