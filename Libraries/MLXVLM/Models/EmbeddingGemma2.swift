// Copyright © 2026 Faux Fox, Inc.
// Portions © 2025 Prince Canuma (MIT), © 2026 Google LLC and Hugging Face (Apache-2.0).
// See LICENSES/EmbeddingGemma2-NOTICE.md for source attribution and license texts.
// Native Swift port of Google/Hugging Face EmbeddingGemma 2 and MLX-VLM.

import Foundation
import MLX
import MLXLMCommon
import MLXNN

struct EmbeddingGemma2Configuration: Decodable, Sendable {
    let modelType: String
    let text: EmbeddingGemma2TextConfiguration
    let vision: Gemma4VisionConfiguration
    let audio: EmbeddingGemma2AudioConfiguration
    let imageToken: Int
    let audioToken: Int
    let videoToken: Int
    let beginImage: Int
    let endImage: Int
    let beginAudio: Int
    let endAudio: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case text = "text_config"
        case vision = "vision_config"
        case audio = "audio_config"
        case imageToken = "image_token_id"
        case audioToken = "audio_token_id"
        case videoToken = "video_token_id"
        case beginImage = "boi_token_id"
        case endImage = "eoi_token_id"
        case beginAudio = "boa_token_id"
        case endAudio = "eoa_token_index"
    }

    func validate() throws {
        try text.validate()
        guard text.hiddenSize == 512, text.hiddenLayers == 24, text.intermediateSize == 2048,
            text.heads == 4, text.kvHeads == 2, text.headDim == 256,
            text.perLayerWidth == 512, text.vocabularySize == 262144,
            text.slidingWindow == 512, !text.attentionBias,
            text.rotary["sliding_attention"]?.theta == 10_000,
            text.rotary["full_attention"]?.theta == 1_000_000
        else {
            throw EmbeddingGemma2Error.invalidConfiguration(
                "requires the EmbeddingGemma 2 text topology")
        }
        for i in 0 ..< 24 {
            let global = (i + 1) % 6 == 0
            let layer = text.perLayer[String(format: "%02d", i)]
            guard text.layerTypes[i] == (global ? "full_attention" : "sliding_attention"),
                (layer?.headDim ?? text.headDim) == (global ? 512 : 256),
                (layer?.kvHeads ?? text.kvHeads) == (global ? 1 : 2),
                (layer?.heads ?? text.heads) == 4
            else { throw EmbeddingGemma2Error.invalidConfiguration("unsupported text layer \(i)") }
        }
        guard audio.modelType == "gemma4_audio", audio.hiddenSize == 1024,
            audio.numHiddenLayers == 12, audio.numAttentionHeads == 8,
            audio.subsamplingConvChannels == [128, 32], audio.convKernelSize == 5,
            audio.attentionChunkSize == 12, audio.attentionContextLeft == 13,
            audio.attentionContextRight == 0, audio.attentionLogitCap == 50,
            audio.outputProjDims == 1536, audio.hiddenAct == "silu",
            audio.rmsNormEps.isFinite, audio.rmsNormEps > 0,
            audio.gradientClipping.isFinite, audio.gradientClipping > 0,
            audio.residualWeight == 0.5
        else {
            throw EmbeddingGemma2Error.invalidConfiguration(
                "requires the complete EmbeddingGemma 2 audio topology")
        }
        guard modelType == "embedding_gemma2", vision.modelType == "gemma4_vision",
            vision.patchSize == 16, vision.poolingKernelSize == 3,
            vision.hiddenSize == 768, vision.headDim == 64,
            vision.hiddenLayers == 16, vision.intermediateSize == 3072,
            vision.attentionHeads == 12, vision.keyValueHeads == 12,
            vision.positionEmbeddingSize > 0, vision.rmsNormEps > 0,
            vision.ropeParameters["rope_theta"]?.asFloat() == 100
        else {
            throw EmbeddingGemma2Error.invalidConfiguration(
                "requires the complete EmbeddingGemma 2 vision topology")
        }
        let special = [
            imageToken, audioToken, videoToken, beginImage, endImage, beginAudio, endAudio,
        ]
        guard Set(special).count == special.count,
            special.allSatisfy({ (0 ..< text.vocabularySize).contains($0) })
        else { throw EmbeddingGemma2Error.invalidConfiguration("invalid modality token IDs") }
    }
}

// Fixed component roots let each tower load and release without mutating an MLX module cache.
final class EmbeddingGemma2TextComponent: Module {
    @ModuleInfo(key: "language_model") var encoder: EmbeddingGemma2TextEncoder
    init(_ c: EmbeddingGemma2Configuration) {
        _encoder.wrappedValue = EmbeddingGemma2TextEncoder(c.text)
        super.init()
    }
}

final class EmbeddingGemma2VisionComponent: Module {
    @ModuleInfo(key: "vision_tower") var encoder: EmbeddingGemma2VisionEncoder
    @ModuleInfo(key: "embed_vision") var projection: EmbeddingGemma2MultimodalProjection
    init(_ c: EmbeddingGemma2Configuration) {
        _encoder.wrappedValue = EmbeddingGemma2VisionEncoder(c.vision)
        _projection.wrappedValue = EmbeddingGemma2MultimodalProjection(
            inputSize: c.vision.hiddenSize, hiddenSize: c.text.hiddenSize,
            epsilon: c.vision.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ image: EmbeddingGemma2PreparedImage) throws -> MLXArray {
        projection(try encoder(image))
    }
}

final class EmbeddingGemma2AudioComponent: Module {
    @ModuleInfo(key: "audio_tower") var encoder: EmbeddingGemma2AudioEncoder
    @ModuleInfo(key: "embed_audio") var projection: EmbeddingGemma2MultimodalProjection
    init(_ c: EmbeddingGemma2Configuration) {
        _encoder.wrappedValue = EmbeddingGemma2AudioEncoder(c.audio)
        _projection.wrappedValue = EmbeddingGemma2MultimodalProjection(
            inputSize: c.audio.outputProjDims ?? c.audio.hiddenSize,
            hiddenSize: c.text.hiddenSize, epsilon: c.audio.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ audio: EmbeddingGemma2PreparedAudio) throws -> MLXArray {
        let output = encoder(audio.features, validMask: audio.validMask)
        let valid = output.validMask.asArray(Bool.self)
        let actual = valid.filter { $0 }.count
        guard actual == audio.tokenCount, actual > 0,
            valid.prefix(actual).allSatisfy({ $0 }),
            valid.dropFirst(actual).allSatisfy({ !$0 })
        else {
            throw EmbeddingGemma2Error.featureCount(
                modality: "audio", expected: audio.tokenCount, actual: actual)
        }
        return projection(output.features)[0, ..<actual, 0...]
    }
}
