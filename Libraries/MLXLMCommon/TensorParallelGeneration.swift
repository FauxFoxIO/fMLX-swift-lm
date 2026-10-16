// Copyright © 2026 Faux Fox, Inc.

import Foundation
import MLX

/// The parameters of one generation that every rank must share.
///
/// Prompt, prefill and cache settings decide how many forwards run and with which shapes,
/// so they must match on every rank. Sampling settings only matter on rank 0, since every
/// rank continues with rank 0's token, but they travel along so ranks stay identical.
public struct TensorParallelGenerationRequest: Codable, Sendable, Equatable {
    public var promptTokens: [Int]
    public var maxTokens: Int
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var minP: Float
    public var seed: UInt64?
    public var repetitionPenalty: Float?
    public var repetitionContextSize: Int
    public var prefillStepSize: Int?
    public var maxKVSize: Int?
    public var kvBits: Int?
    public var kvGroupSize: Int
    public var quantizedKVStart: Int

    public init(
        promptTokens: [Int], maxTokens: Int, temperature: Float = 0, topP: Float = 1,
        topK: Int = 0, minP: Float = 0, seed: UInt64? = nil, repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = 20, prefillStepSize: Int? = nil, maxKVSize: Int? = nil,
        kvBits: Int? = nil, kvGroupSize: Int = 64, quantizedKVStart: Int = 0
    ) {
        self.promptTokens = promptTokens
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.seed = seed
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.prefillStepSize = prefillStepSize
        self.maxKVSize = maxKVSize
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
    }

    public var generateParameters: GenerateParameters {
        GenerateParameters(
            maxTokens: maxTokens, maxKVSize: maxKVSize, kvBits: kvBits,
            kvGroupSize: kvGroupSize, quantizedKVStart: quantizedKVStart,
            temperature: temperature, topP: topP, topK: topK, minP: minP,
            repetitionPenalty: repetitionPenalty, repetitionContextSize: repetitionContextSize,
            prefill: .init(stepSize: prefillStepSize), seed: seed)
    }
}

/// Runs generation in lockstep on every rank of a tensor-parallel group.
///
/// Rank 0 leads: it calls ``generate(_:stopTokens:onToken:)`` for each request and
/// ``shutdown()`` at the end. Every other rank calls ``followRequests()``, which runs the
/// same forwards until rank 0 shuts it down.
///
/// Each rank runs its own ``TokenIterator``, which continues with rank 0's sampled token.
/// Rank 0 alone decides when to stop (EOS, a stop string, cancellation) and sends that
/// decision one step behind the tokens, so the pipeline never waits on it. Each rank then
/// runs one extra forward at the end of a request; that keeps the collectives matched.
///
/// The batching ``ConcurrentTextRuntime`` is single-rank: it cannot drive a sharded model.
public final class TensorParallelGenerator {
    public let model: any LanguageModel
    public let group: MLXDistributed.Group

    public var isLeader: Bool { group.rank == 0 }

    private enum Message: Codable {
        case generate(TensorParallelGenerationRequest)
        case shutdown
    }

    /// - Parameters:
    ///   - model: a model sharded with ``ShardableModel/shard(group:)``, or any model when
    ///     `group` has one rank
    ///   - group: the group, defaulting to the one the model was sharded across
    public init(model: any LanguageModel, group: MLXDistributed.Group? = nil) throws {
        guard
            let group = group ?? (model as? any ShardableModel)?.tensorParallelGroup
        else {
            throw TensorParallelError.modelNotShardable(String(describing: type(of: model)))
        }
        self.model = model
        self.group = group
    }

    /// Rank 0: generate for `request` on every rank.
    ///
    /// - Parameters:
    ///   - request: the request, which is sent to the other ranks
    ///   - stopTokens: tokens that end generation, such as EOS; they are not returned
    ///   - onToken: called with each token; return `false` to stop
    /// - Returns: the generated tokens
    @discardableResult
    public func generate(
        _ request: TensorParallelGenerationRequest, stopTokens: Set<Int> = [],
        onToken: (Int) -> Bool = { _ in true }
    ) throws -> [Int] {
        precondition(isLeader, "only rank 0 starts a generation")
        try send(.generate(request))
        return try runLockstep(request) { token in
            !stopTokens.contains(token) && onToken(token)
        }
    }

    /// Rank 0: tell the other ranks to return from ``followRequests()``.
    public func shutdown() throws {
        precondition(isLeader, "only rank 0 shuts the group down")
        try send(.shutdown)
    }

    /// Ranks other than 0: run each request rank 0 starts until it calls ``shutdown()``.
    ///
    /// - Parameter onRequest: called with each request and the tokens this rank generated
    public func followRequests(
        onRequest: (TensorParallelGenerationRequest, [Int]) -> Void = { _, _ in }
    ) throws {
        precondition(!isLeader, "rank 0 leads; it does not follow")
        while true {
            switch try receive() {
            case .shutdown:
                return
            case .generate(let request):
                let tokens = try runLockstep(request) { _ in true }
                onRequest(request, tokens)
            }
        }
    }

    /// The token loop every rank runs. Only rank 0's `leaderDecides` is used.
    private func runLockstep(
        _ request: TensorParallelGenerationRequest, leaderDecides: (Int) -> Bool
    ) throws -> [Int] {
        let input = LMInput(tokens: MLXArray(request.promptTokens.map { Int32($0) }))
        var iterator = try TokenIterator(
            input: input, model: model, parameters: request.generateParameters)

        var tokens = [Int]()
        var stopping = false
        var pendingStop: MLXArray?
        while let token = iterator.next() {
            // the decision about the previous token is ready by now
            if let pendingStop, pendingStop.item(Int32.self) != 0 {
                break
            }
            if !isLeader {
                tokens.append(token)
            } else if !stopping {
                if leaderDecides(token) {
                    tokens.append(token)
                } else {
                    stopping = true
                }
            }
            let stop = MLXDistributed.broadcast(MLXArray(Int32(stopping ? 1 : 0)), group: group)
            asyncEval(stop)
            pendingStop = stop
        }
        // settle the last forward and decision so no collective is left in flight
        eval([iterator.y.tokens] + (pendingStop.map { [$0] } ?? []))
        return tokens
    }

    private func send(_ message: Message) throws {
        let bytes = try JSONEncoder().encode(message)
        _ = exchange(bytes)
    }

    private func receive() throws -> Message {
        let bytes = exchange(nil)
        do {
            return try JSONDecoder().decode(Message.self, from: bytes)
        } catch {
            throw TensorParallelError.invalidMessage(String(describing: error))
        }
    }

    /// Rank 0 passes the bytes; every rank returns them.
    private func exchange(_ bytes: Data?) -> Data {
        let count = MLXDistributed.broadcast(
            MLXArray(Int32(bytes?.count ?? 0)), group: group
        ).item(Int32.self)
        let payload =
            bytes.map { MLXArray(Array($0)) } ?? MLXArray.zeros([Int(count)], dtype: .uint8)
        let received = MLXDistributed.broadcast(payload, group: group)
        return Data(received.asArray(UInt8.self))
    }
}
