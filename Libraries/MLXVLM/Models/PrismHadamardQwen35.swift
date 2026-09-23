// Copyright © 2026 Faux Fox.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public enum PrismHadamardVLMArtifactError: Error, Equatable, LocalizedError {
    case unsupportedSchema
    case invalidConfiguration(String)
    case missingManifest(String)
    case invalidManifest(String)
    case incompatibleModule(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedSchema: "Unsupported Prism Hadamard VLM artifact schema."
        case .invalidConfiguration(let reason):
            "Invalid Prism Hadamard VLM configuration: \(reason)"
        case .missingManifest(let name): "Missing Prism Hadamard manifest: \(name)"
        case .invalidManifest(let reason): "Invalid Prism Hadamard manifest: \(reason)"
        case .incompatibleModule(let path):
            "Prism Hadamard module is missing or incompatible: \(path)"
        }
    }
}

public struct PrismHadamardQwen35VLMConfiguration: Decodable, Sendable,
    ModelConfigurationValidating
{
    struct Components: Decodable, Sendable {
        let text: Bool
        let vision: Bool
        let mtp: Bool
    }

    struct Quantization: Decodable, Sendable {
        let bits: Int
        let groupSize: Int
        let mode: String

        enum CodingKeys: String, CodingKey {
            case bits, mode
            case groupSize = "group_size"
        }
    }

    struct ModuleRecord: Decodable, Sendable {
        let path: String
        let block: Int
        let embedding: Bool
        let dtype: String
    }

    let qwen: Qwen35Configuration
    let schemaVersion: Int
    let modelType: String
    let baseModelType: String
    let components: Components
    let tensorNamespace: String
    let gdnActivationLayout: String
    let hadamardConfig: String
    let quantization: Quantization
    let modules: [ModuleRecord]

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case modelType = "model_type"
        case baseModelType = "base_model_type"
        case components
        case tensorNamespace = "tensor_namespace"
        case gdnActivationLayout = "gdn_activation_layout"
        case hadamardConfig = "hadamard_config"
        case quantization, modules
    }

    public init(from decoder: Decoder) throws {
        qwen = try Qwen35Configuration(from: decoder)
        let values = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try values.decode(Int.self, forKey: .schemaVersion)
        modelType = try values.decode(String.self, forKey: .modelType)
        baseModelType = try values.decode(String.self, forKey: .baseModelType)
        components = try values.decode(Components.self, forKey: .components)
        tensorNamespace = try values.decode(String.self, forKey: .tensorNamespace)
        gdnActivationLayout = try values.decode(String.self, forKey: .gdnActivationLayout)
        hadamardConfig = try values.decode(String.self, forKey: .hadamardConfig)
        quantization = try values.decode(Quantization.self, forKey: .quantization)
        modules = try values.decode([ModuleRecord].self, forKey: .modules)
    }

    public func validateModelConfiguration() throws {
        guard schemaVersion == 2, modelType == "prism_hadamard_qwen35",
            baseModelType == "qwen3_5"
        else { throw PrismHadamardVLMArtifactError.unsupportedSchema }
        guard components.text, components.vision, !components.mtp,
            qwen.textConfiguration.mtpNumHiddenLayers == 0
        else {
            throw PrismHadamardVLMArtifactError.invalidConfiguration(
                "text and vision must be present and MTP must be absent")
        }
        let text = qwen.textConfiguration
        guard text.hiddenSize == 5_120, text.hiddenLayers == 64,
            text.intermediateSize == 17_408, text.attentionHeads == 24,
            text.kvHeads == 4, text.headDim == 256,
            text.linearNumValueHeads == 48, text.linearNumKeyHeads == 16,
            text.linearKeyHeadDim == 128, text.linearValueHeadDim == 128,
            text.fullAttentionInterval == 4, text.vocabularySize == 248_320
        else {
            throw PrismHadamardVLMArtifactError.invalidConfiguration(
                "unsupported Bonsai Qwen 3.5 dimensions")
        }
        guard tensorNamespace == "mlx-vlm-qwen3_5", gdnActivationLayout == "grouped" else {
            throw PrismHadamardVLMArtifactError.invalidConfiguration("unsupported tensor layout")
        }
        guard quantization.bits == 2, quantization.groupSize == 128,
            quantization.mode == "affine"
        else {
            throw PrismHadamardVLMArtifactError.invalidConfiguration(
                "the language model must use affine 2-bit group-128 weights")
        }
        guard hadamardConfig == URL(fileURLWithPath: hadamardConfig).lastPathComponent,
            !hadamardConfig.isEmpty
        else { throw PrismHadamardVLMArtifactError.invalidConfiguration("invalid manifest path") }
        guard modules.count == 402, Set(modules.map(\.path)).count == modules.count else {
            throw PrismHadamardVLMArtifactError.invalidConfiguration(
                "the artifact must describe 402 unique transformed modules")
        }
        for module in modules {
            let components = module.path.split(separator: ".", omittingEmptySubsequences: false)
            guard !components.isEmpty,
                components.allSatisfy({
                    !$0.isEmpty && $0.allSatisfy { $0.isLetter || $0.isNumber || $0 == "_" }
                }), module.block == 1024, module.dtype == "float16"
            else {
                throw PrismHadamardVLMArtifactError.invalidConfiguration(
                    "unsupported module record at \(module.path)")
            }
        }
        guard modules.filter(\.embedding).map(\.path) == ["model.embed_tokens"] else {
            throw PrismHadamardVLMArtifactError.invalidConfiguration(
                "exactly model.embed_tokens must use the inverse transform")
        }
    }
}

private struct PrismHadamardVLMManifest: Decodable {
    let version: Int
    let blockSize: Int
    let transform: String
    let axis: String
    let signMode: String
    let signWidths: [Int]
    let signValues: [Float]
    let weightNames: [String]
    let inverseWeightNames: [String]
    let gdnVGrouped: Bool

    enum CodingKeys: String, CodingKey {
        case version = "prism.hadamard.version"
        case blockSize = "prism.hadamard.block_size"
        case transform = "prism.hadamard.transform"
        case axis = "prism.hadamard.axis"
        case signMode = "prism.hadamard.sign_mode"
        case signWidths = "prism.hadamard.sign_widths"
        case signValues = "prism.hadamard.sign_values"
        case weightNames = "prism.hadamard.weight_names"
        case inverseWeightNames = "prism.hadamard.inverse_weight_names"
        case gdnVGrouped = "prism.hadamard.gdn_v_grouped"
    }
}

public final class PrismHadamardQwen35VLM: Qwen35, ModelArtifactPreparing,
    LoadedWeightsPreparing, InferenceArtifactIdentityProviding
{
    private let artifactConfiguration: PrismHadamardQwen35VLMConfiguration
    private var preparedDirectory: URL?
    private var records = [PrismHadamardQwen35VLMConfiguration.ModuleRecord]()
    private var contracts = [Int: HadamardTransformContract]()
    private var signValidationFailure: PrismHadamardVLMArtifactError?
    public private(set) var transformContractRevision = "unprepared"

    public init(_ configuration: PrismHadamardQwen35VLMConfiguration) {
        artifactConfiguration = configuration
        super.init(configuration.qwen)
    }

    public func prepareArtifact(in modelDirectory: URL) throws {
        let directory = modelDirectory.standardizedFileURL
        if preparedDirectory == directory { return }
        let url = directory.appendingPathComponent(artifactConfiguration.hadamardConfig)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw PrismHadamardVLMArtifactError.missingManifest(url.lastPathComponent)
        }
        let manifest: PrismHadamardVLMManifest
        do {
            manifest = try JSONDecoder().decode(
                PrismHadamardVLMManifest.self, from: Data(contentsOf: url))
        } catch {
            throw PrismHadamardVLMArtifactError.invalidManifest(error.localizedDescription)
        }
        try validate(manifest)
        var offset = 0
        var loaded = [Int: HadamardTransformContract]()
        for width in manifest.signWidths {
            let signs = Array(manifest.signValues[offset ..< offset + width])
            loaded[width] = HadamardTransformContract(
                identifier:
                    "prism-hadamard-v\(manifest.version)-b\(manifest.blockSize)-w\(width)-\(prismVLMStableSignHash(signs))",
                blockSize: manifest.blockSize, signs: MLXArray(signs), direction: .forward)
            offset += width
        }
        records = artifactConfiguration.modules
        contracts = loaded
        transformContractRevision =
            "prism-hadamard-v\(manifest.version)-b\(manifest.blockSize)-"
            + prismVLMStableSignHash(manifest.signValues)
        preparedDirectory = directory
    }

    public override func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        if !contracts.isEmpty {
            for record in artifactConfiguration.modules {
                let key = "language_model.\(record.path).signs"
                guard let supplied = weights[key] else { continue }
                let width = supplied.dim(0)
                guard supplied.ndim == 1, let expected = contracts[width]?.signs else {
                    signValidationFailure = .invalidManifest("invalid sign tensor at \(key)")
                    continue
                }
                if !all(supplied.asType(.float32) .== expected).item(Bool.self) {
                    signValidationFailure = .invalidManifest("sign tensor disagrees at \(key)")
                }
            }
        }
        return super.sanitize(weights: weights).filter { !$0.key.hasSuffix(".signs") }
    }

    public func prepareLoadedWeights() throws {
        if let signValidationFailure { throw signValidationFailure }
        guard preparedDirectory != nil, !records.isEmpty, !contracts.isEmpty else {
            throw PrismHadamardVLMArtifactError.invalidManifest("artifact was not prepared")
        }
        let leaves = Dictionary(uniqueKeysWithValues: leafModules().flattened())
        var replacements = [(String, Module)]()
        replacements.reserveCapacity(records.count)
        for record in records {
            let path = "language_model.\(record.path)"
            guard let module = leaves[path] else {
                throw PrismHadamardVLMArtifactError.incompatibleModule(path)
            }
            if record.embedding {
                guard
                    ObjectIdentifier(type(of: module)) == ObjectIdentifier(QuantizedEmbedding.self),
                    let embedding = module as? QuantizedEmbedding,
                    let forward = contracts[embedding.shape.1]
                else { throw PrismHadamardVLMArtifactError.incompatibleModule(path) }
                replacements.append(
                    (
                        path,
                        HadamardQuantizedEmbedding(
                            embedding,
                            transform: HadamardTransformContract(
                                identifier: forward.identifier, blockSize: forward.blockSize,
                                signs: forward.signs, direction: .inverse))
                    ))
            } else {
                guard ObjectIdentifier(type(of: module)) == ObjectIdentifier(QuantizedLinear.self),
                    let linear = module as? QuantizedLinear,
                    let transform = contracts[linear.shape.1]
                else { throw PrismHadamardVLMArtifactError.incompatibleModule(path) }
                replacements.append((path, HadamardQuantizedLinear(linear, transform: transform)))
            }
        }
        update(modules: ModuleChildren.unflattened(replacements))
    }

    private func validate(_ manifest: PrismHadamardVLMManifest) throws {
        guard manifest.version == 1, manifest.blockSize == 1024,
            manifest.transform == "normalized-sylvester-walsh-hadamard",
            manifest.axis == "input-last-dimension", manifest.signMode == "explicit",
            manifest.gdnVGrouped
        else {
            throw PrismHadamardVLMArtifactError.invalidManifest("unsupported transform contract")
        }
        guard Set(manifest.signWidths) == [5_120, 6_144, 17_408],
            Set(manifest.signWidths).count == manifest.signWidths.count,
            manifest.signWidths.allSatisfy({ $0 > 0 && $0.isMultiple(of: manifest.blockSize) }),
            manifest.signValues.count == manifest.signWidths.reduce(0, +),
            manifest.signValues.allSatisfy({ $0 == -1 || $0 == 1 })
        else { throw PrismHadamardVLMArtifactError.invalidManifest("invalid explicit signs") }
        let forward = Set(
            artifactConfiguration.modules.filter { !$0.embedding }
                .map { "language_model.\($0.path).weight" })
        let inverse = Set(
            artifactConfiguration.modules.filter(\.embedding)
                .map { "language_model.\($0.path).weight" })
        guard Set(manifest.weightNames) == forward,
            Set(manifest.inverseWeightNames) == inverse, forward.isDisjoint(with: inverse)
        else { throw PrismHadamardVLMArtifactError.invalidManifest("module manifests disagree") }
    }
}

private func prismVLMStableSignHash(_ signs: [Float]) -> String {
    var hash: UInt64 = 1_469_598_103_934_665_603
    for sign in signs {
        hash ^= sign < 0 ? 0xff : 0x01
        hash &*= 1_099_511_628_211
    }
    return String(hash, radix: 16)
}
