// Copyright © 2026 Faux Fox, Inc.

import Foundation

/// An ordered JSON value for decision states. d1-omni was trained on states written by
/// Python's `json.dumps(state, ensure_ascii=False)`; ``serialized`` writes the same text.
public indirect enum DecisionJSON: Sendable, Hashable {
    case null
    case bool(Bool)
    case integer(Int)
    case number(Double)
    case string(String)
    case array([DecisionJSON])
    /// Members keep their order.
    case object([Member])

    public struct Member: Sendable, Hashable {
        public let key: String
        public let value: DecisionJSON

        public init(_ key: String, _ value: DecisionJSON) {
            self.key = key
            self.value = value
        }
    }

    public var serialized: String {
        var output = ""
        write(to: &output)
        return output
    }

    private func write(to output: inout String) {
        switch self {
        case .null: output += "null"
        case .bool(let value): output += value ? "true" : "false"
        case .integer(let value): output += String(value)
        case .number(let value):
            if value.isNaN {
                output += "NaN"
            } else if value.isInfinite {
                output += value < 0 ? "-Infinity" : "Infinity"
            } else {
                output += "\(value)"
            }
        case .string(let value): Self.quote(value, to: &output)
        case .array(let values):
            output += "["
            for (index, value) in values.enumerated() {
                if index > 0 { output += ", " }
                value.write(to: &output)
            }
            output += "]"
        case .object(let members):
            output += "{"
            for (index, member) in members.enumerated() {
                if index > 0 { output += ", " }
                Self.quote(member.key, to: &output)
                output += ": "
                member.value.write(to: &output)
            }
            output += "}"
        }
    }

    private static func quote(_ value: String, to output: inout String) {
        output += "\""
        for scalar in value.unicodeScalars {
            switch scalar {
            case "\"": output += "\\\""
            case "\\": output += "\\\\"
            case "\n": output += "\\n"
            case "\r": output += "\\r"
            case "\t": output += "\\t"
            case "\u{08}": output += "\\b"
            case "\u{0C}": output += "\\f"
            case _ where scalar.value < 0x20:
                output += String(format: "\\u%04x", scalar.value)
            default: output.unicodeScalars.append(scalar)
            }
        }
        output += "\""
    }
}
