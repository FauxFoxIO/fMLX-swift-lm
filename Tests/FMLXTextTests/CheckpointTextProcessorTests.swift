// Copyright © 2026 Faux Fox.

import FMLXText
import Foundation
import MLX
import MLXLLM
import Testing

private struct Fixture {
    let directory: URL
    init(swappedVocabulary: Bool = false, template: String? = nil, includeTemplate: Bool = true)
        throws
    {
        directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            "FMLXText-\(UUID())")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        do {
            let vocabulary: [String: Int] = [
                "[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "hello": swappedVocabulary ? 4 : 3,
                "world": swappedVocabulary ? 3 : 4, "user": 5, "assistant": 6, "tool": 7,
                "weather": 8, "snow": 9, "🤖": 10, "[END]": 11,
            ]
            try write(
                "tokenizer.json",
                object: [
                    "version": "1.0", "truncation": NSNull(), "padding": NSNull(),
                    "added_tokens": [
                        [
                            "id": 1, "content": "[BOS]", "single_word": false, "lstrip": false,
                            "rstrip": false, "normalized": false, "special": true,
                        ],
                        [
                            "id": 2, "content": "[EOS]", "single_word": false, "lstrip": false,
                            "rstrip": false, "normalized": false, "special": true,
                        ],
                    ],
                    "pre_tokenizer": ["type": "WhitespaceSplit"],
                    "model": ["type": "WordLevel", "vocab": vocabulary, "unk_token": "[UNK]"],
                ])
            try write(
                "config.json",
                object: [
                    "model_type": "llama", "vocab_size": 12, "max_position_embeddings": 128,
                    "eos_token_id": 1,
                ])
            var configuration: [String: Any] = [
                "bos_token": "[BOS]", "eos_token": "[EOS]", "unk_token": "[UNK]",
            ]
            if includeTemplate {
                configuration["chat_template"] =
                    template
                    ?? "{{ bos_token }} {% for message in messages %}{{ message['role'] }} {{ message['content'] }} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
            }
            try write("tokenizer_config.json", object: configuration)
            try write("generation_config.json", object: ["eos_token_id": [11]])
        } catch {
            remove()
            throw error
        }
    }
    func write(_ name: String, object: Any) throws {
        try JSONSerialization.data(withJSONObject: object, options: .sortedKeys).write(
            to: directory.appendingPathComponent(name))
    }
    func remove() { try? FileManager.default.removeItem(at: directory) }
}

@Test func vocabularyAndEOSComeFromTheSelectedCheckpoint() async throws {
    let first = try Fixture()
    defer { first.remove() }
    let second = try Fixture(swappedVocabulary: true)
    defer { second.remove() }
    let a = try await CheckpointTextProcessor.load(directory: first.directory)
    let b = try await CheckpointTextProcessor.load(directory: second.directory)
    #expect(try a.encode("hello world", addSpecialTokens: false) == [3, 4])
    #expect(try b.encode("hello world", addSpecialTokens: false) == [4, 3])
    #expect(a.stopTokenIDs == [2, 11])
    #expect(!a.stopTokenIDs.contains(1))
    #expect(a.tokenizerRevision != b.tokenizerRevision)
    #expect(
        try a.prepareChat(messages: [["role": "user", "content": "hello 🤖"]]) == [1, 5, 3, 10, 6])
    #expect(try a.decode([3, 10]) == "hello 🤖")
    #expect(try a.decode([1, 3, 2], skipSpecialTokens: true) == "hello")
}

@Test func samplingDefaultsComeFromGenerationConfiguration() async throws {
    let fixture = try Fixture()
    defer { fixture.remove() }
    try fixture.write(
        "generation_config.json",
        object: [
            "eos_token_id": [11], "do_sample": true,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        ]
    )
    let sampled = try await CheckpointTextProcessor.load(directory: fixture.directory)
    #expect(sampled.temperature == 1)
    #expect(sampled.topP == 0.95)
    #expect(sampled.topK == 20)

    try fixture.write(
        "generation_config.json",
        object: ["eos_token_id": [11], "do_sample": false, "temperature": 1.0]
    )
    let greedy = try await CheckpointTextProcessor.load(directory: fixture.directory)
    #expect(greedy.temperature == 0)
}

@Test func templateChangesInvalidateCacheAndSidecarWins() async throws {
    let fixture = try Fixture()
    defer { fixture.remove() }
    let a = try await CheckpointTextProcessor.load(directory: fixture.directory)
    try Data("world".utf8).write(
        to: fixture.directory.appendingPathComponent("chat_template.jinja"))
    let b = try await CheckpointTextProcessor.load(directory: fixture.directory)
    #expect(try b.prepareChat(messages: [["role": "user", "content": "hello"]]) == [4])
    #expect(a.chatTemplateRevision != b.chatTemplateRevision)
    #expect(
        try a.cacheIdentity(modelRevision: "weights-a", cacheLayoutRevision: "native")
            != b.cacheIdentity(modelRevision: "weights-a", cacheLayoutRevision: "native"))
    #expect(try a.prepareChat(messages: [["role": "user", "content": "hello"]]) == [1, 5, 3, 6])
}

@Test func namedToolTemplatePreservesStructuredArgumentsAndOptions() async throws {
    let fixture = try Fixture()
    defer { fixture.remove() }
    try fixture.write(
        "tokenizer_config.json",
        object: [
            "eos_token": "[EOS]",
            "chat_template": [
                ["name": "default", "template": "hello"],
                [
                    "name": "tool_use",
                    "template":
                        "{{ tools[0]['function']['name'] }} {{ messages[0]['tool_calls'][0]['function']['arguments']['condition'] }} {% if enable_thinking == false %}world{% endif %}",
                ],
            ],
        ])
    let text = try await CheckpointTextProcessor.load(directory: fixture.directory)
    let messages: [[String: any Sendable]] = [
        [
            "role": "assistant", "content": "",
            "tool_calls": [
                [
                    "function": ["name": "weather", "arguments": ["condition": "snow"]]
                        as [String: any Sendable]
                ]
            ],
        ]
    ]
    let tools: [[String: any Sendable]] = [["type": "function", "function": ["name": "weather"]]]
    #expect(
        try text.prepareChat(
            messages: messages, tools: tools, additionalContext: ["enable_thinking": false]) == [
                8, 9, 4,
            ])
    #expect(try text.prepareChat(messages: messages) == [3])
    #expect(throws: CheckpointTextError.self) {
        try text.prepareChat(messages: messages, additionalContext: ["messages": [] as [String]])
    }
}

@Test func missingTemplateAndInvalidVocabularyFailWithoutFallback() async throws {
    let fixture = try Fixture(includeTemplate: false)
    defer { fixture.remove() }
    let text = try await CheckpointTextProcessor.load(directory: fixture.directory)
    #expect(!text.hasChatTemplate)
    #expect(throws: (any Error).self) {
        try text.prepareChat(messages: [["role": "user", "content": "hello"]])
    }
    #expect(throws: CheckpointTextError.invalidTokenID(12)) { try text.decode([12]) }
    try fixture.write("config.json", object: ["model_type": "llama", "vocab_size": 5])
    await #expect(throws: CheckpointTextError.self) {
        try await CheckpointTextProcessor.load(directory: fixture.directory)
    }
}

@Test func contextAccountingIncludesChatFramingAndNeverTruncates() async throws {
    let fixture = try Fixture()
    defer { fixture.remove() }
    let text = try await CheckpointTextProcessor.load(directory: fixture.directory)
    let tokens = try text.prepareChat(messages: [
        ["role": "user", "content": Array(repeating: "hello", count: 140).joined(separator: " ")]
    ])
    #expect(tokens.count == 143)
    #expect(
        throws: CheckpointTextError.contextExceeded(
            promptTokens: 143, outputTokens: 8, capacity: 128)
    ) {
        try text.validateContext(promptTokenCount: tokens.count, maximumOutputTokens: 8)
    }
    try text.validateContext(promptTokenCount: 120, maximumOutputTokens: 8)
}

@Test func malformedSidecarsAndDuplicateTemplateNamesFailDuringLoad() async throws {
    let fixture = try Fixture()
    defer { fixture.remove() }
    try Data("{broken".utf8).write(
        to: fixture.directory.appendingPathComponent("tokenizer_config.json"))
    await #expect(throws: (any Error).self) {
        try await CheckpointTextProcessor.load(directory: fixture.directory)
    }
    try fixture.write(
        "tokenizer_config.json",
        object: [
            "chat_template": [
                ["name": "default", "template": "hello"], ["name": "default", "template": "world"],
            ]
        ])
    await #expect(throws: CheckpointTextError.self) {
        try await CheckpointTextProcessor.load(directory: fixture.directory)
    }
}

@Test func parallelPreparationAndDecodersKeepIndependentState() async throws {
    let fixture = try Fixture()
    defer { fixture.remove() }
    let text = try await CheckpointTextProcessor.load(directory: fixture.directory)
    try await withThrowingTaskGroup(of: [Int].self) { group in
        for _ in 0 ..< 20 {
            group.addTask {
                try text.prepareChat(messages: [["role": "user", "content": "hello"]])
            }
        }
        for try await tokens in group { #expect(tokens == [1, 5, 3, 6]) }
    }
    let first = text.makeDecoder()
    let second = text.makeDecoder()
    #expect(try first.consume(3) == "hello")
    #expect(try second.consume(4) == "world")
    #expect(try first.consume(10) == " 🤖")
    #expect(try second.consume(3) == " hello")
}

@Test func installedCheckpointTokenizationProbe() async throws {
    guard let path = ProcessInfo.processInfo.environment["FMLX_TOKENIZER_CHECKPOINT"] else {
        return
    }
    let text = try await CheckpointTextProcessor.load(directory: URL(fileURLWithPath: path))
    let input = "Hello, Bright Eyes. 你好 👋\nSwift: let answer = 42"
    let tokens = try text.encode(input, addSpecialTokens: false)
    if let expected = ProcessInfo.processInfo.environment["FMLX_EXPECTED_RAW_TOKENS"] {
        let reference = try JSONDecoder().decode([Int].self, from: Data(expected.utf8))
        #expect(tokens == reference)
    }
    #expect(try text.decode(tokens) == input)
    let stream = text.makeDecoder()
    var streamed = ""
    for token in tokens { streamed += try stream.consume(token) ?? "" }
    #expect(streamed == input)
    let prompt = try text.prepareChat(
        messages: [["role": "user", "content": input]],
        additionalContext: ["enable_thinking": false])
    #expect(prompt.count > tokens.count)
    #expect(!text.stopTokenIDs.isEmpty)
    print(
        "fMLX checkpoint tokenizer: \(path); text tokens=\(tokens.count), chat tokens=\(prompt.count), EOS=\(text.stopTokenIDs.sorted())"
    )
}

@Test func nativeModelLoadsItsOwnTokenizerAndGeneratesFromPreparedChat() async throws {
    let fixture = try Fixture()
    defer { fixture.remove() }
    try fixture.write(
        "config.json",
        object: [
            "model_type": "llama", "vocab_size": 12, "max_position_embeddings": 128,
            "hidden_size": 8, "num_hidden_layers": 1, "intermediate_size": 16,
            "num_attention_heads": 2, "num_key_value_heads": 2, "rms_norm_eps": 0.00001,
            "tie_word_embeddings": true,
        ])
    // A tiny real MLX checkpoint exercises loading and inference without model downloads.
    do {
        let configuration = try JSONDecoder().decode(
            LlamaConfiguration.self,
            from: Data(contentsOf: fixture.directory.appendingPathComponent("config.json")))
        let model = LlamaModel(configuration)
        eval(model)
        try save(
            arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
            url: fixture.directory.appendingPathComponent("model.safetensors"))
    }
    let loaded = try await NativeTextModel.load(
        directory: fixture.directory, modelRevision: "test-weights",
        configuration: .init(
            memoryBudgetBytes: 16_000_000, prefixCacheBytes: 1_000_000, workingMemoryBytes: 100_000,
            maxPromptTokens: 128, maxOutputTokens: 8))
    do {
        let request = try loaded.prepareRequest(
            messages: [["role": "user", "content": "hello"]], maximumOutputTokens: 4,
            prefixTokenCount: 3)
        #expect(request.tokens == [1, 5, 3, 6])
        #expect(request.stopTokenIDs == [2, 11])
        #expect(request.cacheIdentity == loaded.cacheIdentity)
        let automaticPrefix = try loaded.prepareRequest(
            messages: [["role": "user", "content": "hello"]],
            maximumOutputTokens: 4,
            cachePromptPrefix: true)
        #expect(automaticPrefix.prefixTokenCount == automaticPrefix.tokens.count - 1)
        let generation = try await loaded.runtime.generate(request)
        var tokens: [Int] = []
        var finished = false
        for try await event in generation.events {
            switch event {
            case .token(let token): tokens.append(token)
            case .finished: finished = true
            default: break
            }
        }
        #expect(finished)
        #expect(tokens.count <= 4)
        _ = try loaded.text.decode(tokens)
        var textCompleted = false
        for try await event in loaded.generateText(request: request) {
            if case .completed(let input, let output, _, _) = event {
                #expect(input == request.tokens.count)
                #expect(output <= 4)
                textCompleted = true
            }
        }
        #expect(textCompleted)
        #expect(await loaded.runtime.status().activeRequests == 0)
        await loaded.runtime.shutdown()
    } catch {
        await loaded.runtime.shutdown()
        throw error
    }
}
