import Foundation
import MLX
import MLXNN

/// One static-shaped piece of a hybrid model's single-token decode graph.
///
/// Full-attention cache writes split segments because the cache grows at every
/// token and cannot be mutated inside an MLX compiled function. Linear-layer
/// state is passed explicitly through the compiled function instead.
package struct CompiledDecodeSegment: Sendable, Equatable {
    package var attentionPostLayer: Int?
    package var linearLayers: [Int]
    package var attentionPreLayer: Int?

    package init(
        attentionPostLayer: Int? = nil,
        linearLayers: [Int] = [],
        attentionPreLayer: Int? = nil
    ) {
        self.attentionPostLayer = attentionPostLayer
        self.linearLayers = linearLayers
        self.attentionPreLayer = attentionPreLayer
    }

    /// First linear-state input, after the hidden state and an optional
    /// `[attention, gate]` pair.
    package var stateInputOffset: Int { attentionPostLayer == nil ? 1 : 3 }

    /// First `[queries, gate, keys, values]` output after the hidden state and
    /// two updated state tensors per linear layer.
    package var attentionOutputOffset: Int { 1 + 2 * linearLayers.count }

    /// Every decoder layer this segment runs, in order. The weights of these
    /// layers are the compile state of the segment's trace.
    package var layerIndices: [Int] {
        (attentionPostLayer.map { [$0] } ?? []) + linearLayers
            + (attentionPreLayer.map { [$0] } ?? [])
    }

    /// Build the maximal segments separated by full-attention cache writes.
    package static func schedule(linearLayers: [Bool]) -> [Self] {
        var segments: [Self] = []
        var current = Self()
        for (index, isLinear) in linearLayers.enumerated() {
            if isLinear {
                current.linearLayers.append(index)
            } else {
                current.attentionPreLayer = index
                segments.append(current)
                current = Self(attentionPostLayer: index)
            }
        }
        segments.append(current)
        return segments
    }
}

/// Lazily compiles and retains one `CompiledTrace` per decode segment.
///
/// Models create this with their schedule; compilation stays lazy because
/// weights load after initialization. Each segment declares the modules its
/// body reads, so the trace sees current weights rather than the ones it was
/// first traced with.
package final class CompiledDecodeSegmentCache<Owner: Module>: CompiledTraceInvalidating {

    /// Runs segment `index` over the flat argument list.
    package typealias Body = @Sendable (Owner, Int, [MLXArray]) -> [MLXArray]

    /// Returns the modules whose weights segment `index` reads: the layers in
    /// `layerIndices`, plus anything else the body touches such as the
    /// embedding or the final norm.
    package typealias StateProvider = @Sendable (Owner, Int) -> [Module]

    private let traces: [CompiledTrace<Owner>]

    package init(count: Int, state: @escaping StateProvider, body: @escaping Body) {
        precondition(count > 0, "compiled decode requires at least one segment")
        self.traces = (0 ..< count).map { index in
            CompiledTrace<Owner>(
                state: { state($0, index) },
                body: { owner, arguments in body(owner, index, arguments) })
        }
    }

    package var compiledCount: Int {
        traces.filter(\.isCompiled).count
    }

    package func callAsFunction(
        _ owner: Owner, at index: Int, _ arguments: [MLXArray]
    ) -> [MLXArray] {
        traces[index](owner, arguments)
    }

    /// Drops every segment trace. See `CompiledTrace.invalidate()`.
    package func invalidate() {
        traces.forEach { $0.invalidate() }
    }
}

// MARK: - Attention-split decoder layers

/// A decoder layer whose single-token step splits around its KV-cache write.
///
/// ``CompiledAttentionDecoder`` traces ``decodeProjections(_:)`` and
/// ``decodeOutput(_:attention:)``, which may read only their arguments and
/// the layer's weights. ``decodeAttention(_:cache:)`` runs eagerly: rope
/// offsets move every token and the cache write cannot live in a trace.
package protocol AttentionSplitDecoderLayer: Module {
    /// Layer input to the pre-rope tensors the attention step needs.
    func decodeProjections(_ x: MLXArray) -> [MLXArray]

    /// Rope, cache update, and attention over the cache, before the output
    /// projection.
    func decodeAttention(_ projections: [MLXArray], cache: KVCache) -> MLXArray

    /// Layer input and attention result to the layer output.
    func decodeOutput(_ x: MLXArray, attention: MLXArray) -> MLXArray
}

/// A model that can switch its compiled single-token decode path off, for
/// comparisons against the eager path.
package protocol CompiledDecodeConfigurable: AnyObject {
    var compiledDecodeEnabled: Bool { get set }
}

/// Compiled batch-1, single-token decode for a stack of
/// ``AttentionSplitDecoderLayer``s.
///
/// Segment 0 runs the embedding and the first layer's projections; segment
/// `i` runs layer `i - 1`'s output half and layer `i`'s projections; the last
/// segment runs the final layer's output half and the model head. Only the
/// attention steps between segments run eagerly, so any cache type works.
package final class CompiledAttentionDecoder<Owner: Module>: CompiledTraceInvalidating {
    package typealias Layers = @Sendable (Owner) -> [any AttentionSplitDecoderLayer]

    private let layers: Layers
    private let segments: CompiledDecodeSegmentCache<Owner>

    /// - Parameters:
    ///   - layerCount: the number of decoder layers
    ///   - layers: the decoder layers, in order
    ///   - embed: token ids to the first layer's input
    ///   - head: the last layer's output to the model output
    ///   - embedState: the modules `embed` reads
    ///   - headState: the modules `head` reads
    package init(
        layerCount: Int,
        layers: @escaping Layers,
        embed: @escaping @Sendable (Owner, MLXArray) -> MLXArray,
        head: @escaping @Sendable (Owner, MLXArray) -> MLXArray,
        embedState: @escaping @Sendable (Owner) -> [Module],
        headState: @escaping @Sendable (Owner) -> [Module]
    ) {
        self.layers = layers
        self.segments = CompiledDecodeSegmentCache(
            count: layerCount + 1,
            state: { owner, index in
                let stack = layers(owner)
                return (index == 0 ? embedState(owner) : [stack[index - 1]])
                    + (index < stack.count ? [stack[index]] : headState(owner))
            },
            body: { owner, index, arguments in
                let stack = layers(owner)
                let h =
                    index == 0
                    ? embed(owner, arguments[0])
                    : stack[index - 1].decodeOutput(arguments[0], attention: arguments[1])
                if index < stack.count {
                    return [h] + stack[index].decodeProjections(h)
                }
                return [head(owner, h)]
            })
    }

    package var compiledCount: Int { segments.compiledCount }

    /// One decode step, or nil when `inputs` is not a single token at batch
    /// one or the caches would need a mask; the caller then runs eagerly.
    package func callAsFunction(
        _ owner: Owner, inputs: MLXArray, cache: [KVCache]?
    ) -> MLXArray? {
        guard inputs.ndim == 2, inputs.dim(0) == 1, inputs.dim(1) == 1,
            let cache, let first = cache.first,
            case .none = createAttentionMask(h: inputs, cache: first)
        else { return nil }
        let stack = layers(owner)
        guard cache.count == stack.count else { return nil }

        var arguments = [inputs]
        for (index, layer) in stack.enumerated() {
            let outputs = segments(owner, at: index, arguments)
            let attention = layer.decodeAttention(Array(outputs[1...]), cache: cache[index])
            arguments = [outputs[0], attention]
        }
        return segments(owner, at: stack.count, arguments)[0]
    }

    package func invalidate() {
        segments.invalidate()
    }
}
