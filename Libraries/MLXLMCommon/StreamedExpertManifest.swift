// Copyright © 2026 Faux Fox.

import Foundation

/// A portable description of expert weights that can be streamed independently
/// of the dense model weights.
public struct StreamedExpertManifest: Codable, Equatable, Sendable {
    public static let currentFormatVersion = 1

    public struct Model: Codable, Equatable, Sendable {
        public let id: String
        public let revision: String

        public init(id: String, revision: String) {
            self.id = id
            self.revision = revision
        }
    }

    public struct Source: Codable, Equatable, Hashable, Sendable {
        /// Absolute byte offset in `file`'s safetensors data section.
        ///
        /// This is the header-relative value from safetensors `data_offsets`,
        /// not a delta from a preceding expert slice.
        public let file: String
        public let offset: UInt64
        public let length: UInt64

        public init(file: String, offset: UInt64, length: UInt64) {
            self.file = file
            self.offset = offset
            self.length = length
        }
    }

    public struct Integrity: Codable, Equatable, Sendable {
        public enum Algorithm: String, Codable, Sendable {
            case sha256
        }

        public let algorithm: Algorithm
        public let digest: String

        public init(algorithm: Algorithm = .sha256, digest: String) {
            self.algorithm = algorithm
            self.digest = digest
        }
    }

    public struct Quantization: Codable, Equatable, Sendable {
        public let bits: Int
        public let groupSize: Int
        public let mode: String

        public init(bits: Int, groupSize: Int, mode: String) {
            self.bits = bits
            self.groupSize = groupSize
            self.mode = mode
        }

        enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
            case mode
        }
    }

    public struct ExpertSlice: Codable, Equatable, Sendable {
        public enum Component: String, Codable, Hashable, Sendable {
            case weight
            case scales
            case biases
        }

        public let index: Int
        public let component: Component
        /// Axis in `shape` that indexes experts. Streamed expert tensors use zero.
        public let expertAxis: Int
        public let source: Source
        public let dtype: String
        public let quantization: Quantization?
        public let shape: [Int]
        public let integrity: Integrity

        public init(
            index: Int,
            component: Component = .weight,
            expertAxis: Int = 0,
            source: Source,
            dtype: String,
            quantization: Quantization? = nil,
            shape: [Int],
            integrity: Integrity
        ) {
            self.index = index
            self.component = component
            self.expertAxis = expertAxis
            self.source = source
            self.dtype = dtype
            self.quantization = quantization
            self.shape = shape
            self.integrity = integrity
        }

        enum CodingKeys: String, CodingKey {
            case index
            case component
            case expertAxis = "expert_axis"
            case source
            case dtype
            case quantization
            case shape
            case integrity
        }
    }

    public struct Projection: Codable, Equatable, Sendable {
        public let name: String
        public let experts: [ExpertSlice]

        public init(name: String, experts: [ExpertSlice]) {
            self.name = name
            self.experts = experts
        }
    }

    public struct Layer: Codable, Equatable, Sendable {
        public let index: Int
        public let projections: [Projection]

        public init(index: Int, projections: [Projection]) {
            self.index = index
            self.projections = projections
        }
    }

    /// A tensor that must remain loaded while experts are streamed on demand.
    public struct DenseTensor: Codable, Equatable, Sendable {
        public let name: String
        public let source: Source
        public let dtype: String
        public let quantization: Quantization?
        public let shape: [Int]
        public let integrity: Integrity

        public init(
            name: String,
            source: Source,
            dtype: String,
            quantization: Quantization? = nil,
            shape: [Int],
            integrity: Integrity
        ) {
            self.name = name
            self.source = source
            self.dtype = dtype
            self.quantization = quantization
            self.shape = shape
            self.integrity = integrity
        }
    }

    public let formatVersion: Int
    public let model: Model
    public let layers: [Layer]
    public let denseResidency: [DenseTensor]

    public init(
        formatVersion: Int = StreamedExpertManifest.currentFormatVersion,
        model: Model,
        layers: [Layer],
        denseResidency: [DenseTensor]
    ) {
        self.formatVersion = formatVersion
        self.model = model
        self.layers = layers
        self.denseResidency = denseResidency
    }

    enum CodingKeys: String, CodingKey {
        case formatVersion = "format_version"
        case model
        case layers
        case denseResidency = "dense_residency"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            formatVersion: try container.decode(Int.self, forKey: .formatVersion),
            model: try container.decode(Model.self, forKey: .model),
            layers: try container.decode([Layer].self, forKey: .layers),
            denseResidency: try container.decode([DenseTensor].self, forKey: .denseResidency)
        )
        try validate()
    }

    /// Validates a manifest before it is used to resolve files or allocate buffers.
    public func validate() throws {
        guard formatVersion == Self.currentFormatVersion else {
            throw StreamedExpertManifestError.unsupportedFormatVersion(formatVersion)
        }
        try validateIdentifier(model.id, field: "model.id")
        try validateIdentifier(model.revision, field: "model.revision")

        var layerIndices = Set<Int>()
        var sourceRanges: [String: [SourceRange]] = [:]

        for layer in layers {
            guard layer.index >= 0 else {
                throw StreamedExpertManifestError.negativeLayerIndex(layer.index)
            }
            guard layerIndices.insert(layer.index).inserted else {
                throw StreamedExpertManifestError.duplicateLayerIndex(layer.index)
            }

            var projectionNames = Set<String>()
            for projection in layer.projections {
                try validateIdentifier(projection.name, field: "projection.name")
                guard projectionNames.insert(projection.name).inserted else {
                    throw StreamedExpertManifestError.duplicateProjection(
                        layer: layer.index, name: projection.name)
                }

                var expertSlices = Set<ExpertSliceKey>()
                for expert in projection.experts {
                    guard expert.index >= 0 else {
                        throw StreamedExpertManifestError.negativeExpertIndex(
                            layer: layer.index, projection: projection.name, index: expert.index)
                    }
                    let key = ExpertSliceKey(index: expert.index, component: expert.component)
                    guard expertSlices.insert(key).inserted else {
                        throw StreamedExpertManifestError.duplicateExpertSlice(
                            layer: layer.index, projection: projection.name, index: expert.index,
                            component: expert.component)
                    }
                    try validate(
                        source: expert.source,
                        dtype: expert.dtype,
                        quantization: expert.quantization,
                        shape: expert.shape,
                        integrity: expert.integrity,
                        expertIndex: expert.index,
                        expertAxis: expert.expertAxis,
                        sourceRanges: &sourceRanges)
                }
            }
        }

        var denseNames = Set<String>()
        for tensor in denseResidency {
            try validateIdentifier(tensor.name, field: "dense_residency.name")
            guard denseNames.insert(tensor.name).inserted else {
                throw StreamedExpertManifestError.duplicateDenseTensor(tensor.name)
            }
            try validate(
                source: tensor.source,
                dtype: tensor.dtype,
                quantization: tensor.quantization,
                shape: tensor.shape,
                integrity: tensor.integrity,
                expertIndex: nil,
                expertAxis: nil,
                sourceRanges: &sourceRanges)
        }

        for (file, ranges) in sourceRanges {
            let sorted = ranges.sorted { $0.offset < $1.offset }
            for (previous, current) in zip(sorted, sorted.dropFirst()) {
                if current.offset < previous.end {
                    throw StreamedExpertManifestError.overlappingSourceRanges(file: file)
                }
            }
        }
    }

    private func validate(
        source: Source,
        dtype: String,
        quantization: Quantization?,
        shape: [Int],
        integrity: Integrity,
        expertIndex: Int?,
        expertAxis: Int?,
        sourceRanges: inout [String: [SourceRange]]
    ) throws {
        try validateSource(source)
        try validateIdentifier(dtype, field: "dtype")
        try validateShape(shape)
        try validateIntegrity(integrity)
        if let expertIndex, let expertAxis {
            guard expertAxis == 0 else {
                throw StreamedExpertManifestError.expertAxisMustBeOuter(expertAxis)
            }
            guard expertIndex < shape[expertAxis] else {
                throw StreamedExpertManifestError.expertIndexOutOfBounds(
                    index: expertIndex, axis: expertAxis, limit: shape[expertAxis])
            }
        }
        if let quantization {
            try validateQuantization(quantization)
        }

        let (end, overflow) = source.offset.addingReportingOverflow(source.length)
        guard !overflow else {
            throw StreamedExpertManifestError.sourceRangeOverflow(file: source.file)
        }
        sourceRanges[source.file, default: []].append(.init(offset: source.offset, end: end))
    }

    private func validateSource(_ source: Source) throws {
        guard source.length > 0 else {
            throw StreamedExpertManifestError.emptySourceRange(file: source.file)
        }
        guard isSafeFileName(source.file) else {
            throw StreamedExpertManifestError.unsafeSourceFile(source.file)
        }
    }

    private func validateShape(_ shape: [Int]) throws {
        guard !shape.isEmpty else { throw StreamedExpertManifestError.emptyShape }
        var elementCount = 1
        for dimension in shape {
            guard dimension > 0 else {
                throw StreamedExpertManifestError.invalidShapeDimension(dimension)
            }
            let (next, overflow) = elementCount.multipliedReportingOverflow(by: dimension)
            guard !overflow else { throw StreamedExpertManifestError.shapeElementCountOverflow }
            elementCount = next
        }
    }

    private func validateIntegrity(_ integrity: Integrity) throws {
        switch integrity.algorithm {
        case .sha256:
            guard integrity.digest.count == 64,
                integrity.digest.allSatisfy({ $0.isHexDigit })
            else {
                throw StreamedExpertManifestError.invalidSHA256Digest(integrity.digest)
            }
        }
    }

    private func validateQuantization(_ quantization: Quantization) throws {
        guard (1 ... 32).contains(quantization.bits) else {
            throw StreamedExpertManifestError.invalidQuantizationBits(quantization.bits)
        }
        guard quantization.groupSize > 0 else {
            throw StreamedExpertManifestError.invalidQuantizationGroupSize(quantization.groupSize)
        }
        try validateIdentifier(quantization.mode, field: "quantization.mode")
    }

    private func validateIdentifier(_ value: String, field: String) throws {
        guard !value.isEmpty,
            value.unicodeScalars.allSatisfy({
                !CharacterSet.whitespacesAndNewlines.contains($0)
                    && !CharacterSet.controlCharacters.contains($0)
            })
        else {
            throw StreamedExpertManifestError.invalidIdentifier(field: field, value: value)
        }
    }

    private func isSafeFileName(_ file: String) -> Bool {
        guard !file.isEmpty, file != ".", file != "..", !file.hasPrefix("/"),
            !file.contains("/"), !file.contains("\\"),
            !file.unicodeScalars.contains(where: { $0.value == 0 })
        else {
            return false
        }
        return file.unicodeScalars.allSatisfy {
            !CharacterSet.controlCharacters.contains($0)
                && !CharacterSet.whitespacesAndNewlines.contains($0)
        }
    }

    private struct SourceRange {
        let offset: UInt64
        let end: UInt64
    }

    private struct ExpertSliceKey: Hashable {
        let index: Int
        let component: ExpertSlice.Component
    }
}

public enum StreamedExpertManifestError: Error, Equatable, Sendable {
    case unsupportedFormatVersion(Int)
    case invalidIdentifier(field: String, value: String)
    case negativeLayerIndex(Int)
    case duplicateLayerIndex(Int)
    case duplicateProjection(layer: Int, name: String)
    case negativeExpertIndex(layer: Int, projection: String, index: Int)
    case duplicateExpertSlice(
        layer: Int, projection: String, index: Int,
        component: StreamedExpertManifest.ExpertSlice.Component)
    case duplicateDenseTensor(String)
    case unsafeSourceFile(String)
    case emptySourceRange(file: String)
    case sourceRangeOverflow(file: String)
    case overlappingSourceRanges(file: String)
    case emptyShape
    case invalidShapeDimension(Int)
    case shapeElementCountOverflow
    case expertAxisMustBeOuter(Int)
    case expertIndexOutOfBounds(index: Int, axis: Int, limit: Int)
    case invalidSHA256Digest(String)
    case invalidQuantizationBits(Int)
    case invalidQuantizationGroupSize(Int)
}
