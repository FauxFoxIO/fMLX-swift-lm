// Copyright © 2026 Faux Fox, Inc.

// Tensor-parallel text generation across the ranks started by scripts/fmlx-launch.py.
// Every rank loads its slice of the model; rank 0 tokenizes, samples and prints.

import FMLXText
import Foundation
import MLX
import MLXLLM
import MLXLMCommon

let usage = """
    usage: fmlx-tp-generate --model DIR [--prompt TEXT] [--chat] [--max-tokens N]
                            [--temperature T] [--top-p P] [--seed S]
                            [--backend any|ring|jaccl]

    Run one copy per rank, normally through scripts/fmlx-launch.py, which sets MLX_RANK,
    the backend's hostfile or device file, and FMLX_BACKEND. Every host needs the model at
    the same path. Raise the GPU wired limit on each Mac first (this tool never does):

        sudo sysctl iogpu.wired_limit_mb=<megabytes>
    """

struct Options {
    var model: URL?
    var prompt = "Write a haiku about two computers sharing one thought."
    var chat = false
    var maxTokens = 128
    var temperature: Float = 0
    var topP: Float = 1
    var seed: UInt64?
    var backend =
        MLXDistributed.Backend(
            rawValue: ProcessInfo.processInfo.environment["FMLX_BACKEND"] ?? "any") ?? .any
}

func fail(_ message: String, code: Int32 = 2) -> Never {
    let rank = ProcessInfo.processInfo.environment["MLX_RANK"] ?? "?"
    FileHandle.standardError.write(Data("fmlx-tp-generate[rank \(rank)]: \(message)\n".utf8))
    exit(code)
}

func parseOptions() -> Options {
    var options = Options()
    var arguments = CommandLine.arguments.dropFirst()
    func value(_ flag: String) -> String {
        guard let value = arguments.popFirst() else { fail("\(flag) needs a value\n\(usage)") }
        return value
    }
    while let argument = arguments.popFirst() {
        switch argument {
        case "--model": options.model = URL(fileURLWithPath: value(argument))
        case "--prompt": options.prompt = value(argument)
        case "--chat": options.chat = true
        case "--max-tokens": options.maxTokens = Int(value(argument)) ?? options.maxTokens
        case "--temperature": options.temperature = Float(value(argument)) ?? 0
        case "--top-p": options.topP = Float(value(argument)) ?? 1
        case "--seed": options.seed = UInt64(value(argument))
        case "--backend":
            let name = value(argument)
            guard let backend = MLXDistributed.Backend(rawValue: name) else {
                fail("unknown backend \(name)\n\(usage)")
            }
            options.backend = backend
        case "-h", "--help":
            print(usage)
            exit(0)
        default:
            fail("unknown argument \(argument)\n\(usage)")
        }
    }
    if options.model == nil { fail("--model is required\n\(usage)") }
    return options
}

func initializeGroup(_ backend: MLXDistributed.Backend) -> MLXDistributed.Group {
    // without a launcher there is no rank, so run as a group of one
    let launched = ProcessInfo.processInfo.environment["MLX_RANK"] != nil
    do {
        return try MLXDistributed.initialize(backend: backend, strict: launched)
    } catch {
        var hint = ""
        if backend == .jaccl {
            hint =
                "\nJACCL needs macOS 26.2 or later, RDMA enabled in recovery (rdma_ctl enable), "
                + "Thunderbolt 5 cables between every pair of Macs, and MLX_IBV_DEVICES / "
                + "MLX_JACCL_COORDINATOR from scripts/fmlx-launch.py. "
                + "Use --backend ring to test on one Mac."
        }
        fail("cannot start the \(backend.rawValue) backend: \(error)\(hint)")
    }
}

let options = parseOptions()
let directory = options.model!
let group = initializeGroup(options.backend)
let isLeader = group.rank == 0
func log(_ message: String) {
    FileHandle.standardError.write(Data("[rank \(group.rank)/\(group.size)] \(message)\n".utf8))
}
if group.size == 1 {
    log("running on one rank; launch with scripts/fmlx-launch.py to shard")
}

do {
    let configuration = try Data(contentsOf: directory.appendingPathComponent("config.json"))
    let base = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configuration)
    let model = try await LLMTypeRegistry.shared.createModel(
        configuration: configuration, modelType: base.modelType)

    let start = Date()
    try await loadWeights(
        modelDirectory: directory, model: model, tensorParallelGroup: group,
        perLayerQuantization: base.perLayerQuantization)
    model.train(false)
    log(
        String(
            format: "loaded %@ in %.1fs, %.2f GB active", base.modelType,
            Date().timeIntervalSince(start), Double(MLX.Memory.activeMemory) / 1e9))

    let generator = try TensorParallelGenerator(model: model, group: group)
    guard isLeader else {
        try generator.followRequests()
        exit(0)
    }

    let text = try await CheckpointTextProcessor.load(directory: directory)
    let prompt =
        options.chat
        ? try text.prepareChat(messages: [["role": "user", "content": options.prompt]])
        : try text.encode(options.prompt)
    let request = TensorParallelGenerationRequest(
        promptTokens: prompt, maxTokens: options.maxTokens, temperature: options.temperature,
        topP: options.topP, seed: options.seed)

    let decoder = text.makeDecoder()
    var firstToken: Date?
    let begin = Date()
    let tokens = try generator.generate(request, stopTokens: text.stopTokenIDs) { token in
        if firstToken == nil { firstToken = Date() }
        if let piece = try? decoder.consume(token) {
            print(piece, terminator: "")
            fflush(stdout)
        }
        return true
    }
    let end = Date()
    print()
    try generator.shutdown()

    let prefill = (firstToken ?? end).timeIntervalSince(begin)
    let decode = end.timeIntervalSince(firstToken ?? end)
    log(
        String(
            format: "prompt %d tokens in %.2fs (%.1f tok/s); generated %d tokens at %.1f tok/s",
            prompt.count, prefill, Double(prompt.count) / max(prefill, 1e-9), tokens.count,
            Double(max(tokens.count - 1, 0)) / max(decode, 1e-9)))
} catch {
    fail("\(error)", code: 1)
}
