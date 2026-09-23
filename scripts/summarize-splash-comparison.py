#!/usr/bin/env python3
"""Validate and summarize normalized same-machine Splash comparison samples."""

import argparse
import collections
import datetime
import hashlib
import json
import math
import statistics
import sys


def load_json(path):
    with open(path, encoding="utf-8") as source:
        return json.load(source)


def load_json_lines(path):
    rows = []
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for line_number, raw_line in enumerate(source, 1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                rows.append(json.loads(raw_line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows, digest.hexdigest()


def require(mapping, keys, context):
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{context}: missing {', '.join(missing)}")


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[rank]


def describe(values):
    if not values:
        return None
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    return {
        "count": len(values),
        "median": median,
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "medianAbsoluteDeviation": statistics.median(deviations),
        "relativeMAD": statistics.median(deviations) / median if median else None,
    }


def drift(values):
    if len(values) < 4:
        return None
    half = len(values) // 2
    first = statistics.median(values[:half])
    last = statistics.median(values[-half:])
    denominator = max(abs(first), abs(last))
    return abs(last - first) / denominator if denominator else 0.0


def validate_session(protocol, session):
    require(session, ["schemaVersion", "protocolId", "sessionId", "host", "models", "engines", "rawEvidence"], "session")
    if session["schemaVersion"] != 1 or session["protocolId"] != protocol["protocolId"]:
        raise ValueError("session schema or protocol ID does not match")
    require(session["models"], ["target", "draft"], "session.models")
    for name in ("target", "draft"):
        require(
            session["models"][name],
            ["modelId", "revision", "sourceWeightSHA256", "conversionProvenance"],
            f"session.models.{name}",
        )
    engine_ids = set()
    for index, engine in enumerate(session["engines"]):
        require(
            engine,
            [
                "id", "revision", "backendVersion", "buildConfiguration",
                "accelerationState", "targetWeightRepresentation",
                "targetArtifactSHA256", "draftWeightRepresentation",
                "draftArtifactSHA256", "command",
            ],
            f"session.engines[{index}]",
        )
        if engine["id"] in engine_ids:
            raise ValueError(f"duplicate engine ID: {engine['id']}")
        if engine["buildConfiguration"] != "Release":
            raise ValueError(f"{engine['id']} is not a Release build")
        engine_ids.add(engine["id"])
    expected = {
        protocol["comparison"]["baselineEngine"],
        protocol["comparison"]["candidateEngine"],
        *protocol["comparison"]["ordinaryAblations"],
    }
    if not expected.issubset(engine_ids):
        raise ValueError(f"session is missing engines: {sorted(expected - engine_ids)}")
    return engine_ids


def validate_sample(row, index, protocol, session, engine_ids, workloads):
    context = f"sample {index}"
    require(
        row,
        [
            "schemaVersion", "protocolId", "sessionId", "engine", "workload", "trial",
            "warmup", "pairIndex", "pairOrder", "valid", "input", "output", "sampler",
            "cache", "timing", "memory", "speculation",
        ],
        context,
    )
    if row["schemaVersion"] != 1 or row["protocolId"] != protocol["protocolId"]:
        raise ValueError(f"{context}: schema or protocol ID does not match")
    if row["sessionId"] != session["sessionId"]:
        raise ValueError(f"{context}: session ID does not match")
    if row["engine"] not in engine_ids:
        raise ValueError(f"{context}: unknown engine {row['engine']}")
    if row["workload"] not in workloads:
        raise ValueError(f"{context}: unknown workload {row['workload']}")
    pair_orders = protocol["measurement"]["pairOrder"]
    expected_order = pair_orders[row["pairIndex"] % len(pair_orders)]
    if row["pairOrder"] != expected_order:
        raise ValueError(f"{context}: pair order does not match the declared alternation")

    workload = workloads[row["workload"]]
    require(row["input"], ["requests", "promptTokenCounts", "renderedTokenSHA256"], f"{context}.input")
    require(row["output"], ["requestedTokensPerRequest", "generatedTokenCounts", "generatedTokenSHA256", "stopReasons"], f"{context}.output")
    require(row["sampler"], ["temperature", "topP", "topK", "seed"], f"{context}.sampler")
    require(row["cache"], ["state", "reusedPrefixTokens"], f"{context}.cache")
    require(
        row["timing"],
        [
            "ttftMilliseconds", "requestTTFTMilliseconds", "prefillTokensPerSecond",
            "decodeTokensPerSecond", "requestDecodeTokensPerSecond",
            "endToEndMilliseconds", "aggregateTokensPerSecond",
        ],
        f"{context}.timing",
    )
    require(row["memory"], ["peakResidentBytes", "steadyResidentBytes", "peakMLXBytes", "cachedMLXBytes"], f"{context}.memory")
    require(row["speculation"], ["targetForwardCount", "draftProposedTokens", "draftAcceptedTokens", "draftRounds", "fallbacks"], f"{context}.speculation")

    requests = workload["requests"]
    if row["input"]["requests"] != requests:
        raise ValueError(f"{context}: request count does not match workload")
    vector_keys = (
        (row["input"], "promptTokenCounts"),
        (row["input"], "renderedTokenSHA256"),
        (row["output"], "generatedTokenCounts"),
        (row["output"], "generatedTokenSHA256"),
        (row["output"], "stopReasons"),
        (row["cache"], "reusedPrefixTokens"),
        (row["timing"], "requestTTFTMilliseconds"),
        (row["timing"], "requestDecodeTokensPerSecond"),
    )
    for owner, key in vector_keys:
        if len(owner[key]) != requests:
            raise ValueError(f"{context}: {key} must contain one value per request")
    if any(value != workload["promptTokens"] for value in row["input"]["promptTokenCounts"]):
        raise ValueError(f"{context}: prompt token count does not match workload")
    if row["output"]["requestedTokensPerRequest"] != workload["outputTokens"]:
        raise ValueError(f"{context}: output token count does not match workload")
    if row["cache"]["state"] != workload["cacheState"]:
        raise ValueError(f"{context}: cache state does not match workload")
    if any(
        value != workload["expectedReusedPrefixTokens"]
        for value in row["cache"]["reusedPrefixTokens"]
    ):
        raise ValueError(f"{context}: reused prefix count does not match workload")
    prompt_hashes = row["input"]["renderedTokenSHA256"]
    if workload["prefixPattern"].startswith("four identical") and len(set(prompt_hashes)) != 1:
        raise ValueError(f"{context}: shared fanout prompts are not identical")
    if workload["prefixPattern"].startswith("four distinct") and len(set(prompt_hashes)) != requests:
        raise ValueError(f"{context}: distinct fanout prompts are not distinct")

    sampling = protocol["sampling"]
    for key in ("temperature", "topP", "topK", "seed"):
        if row["sampler"][key] != sampling[key]:
            raise ValueError(f"{context}: sampler {key} does not match protocol")
    if row["valid"] and any(
        count != workload["outputTokens"] for count in row["output"]["generatedTokenCounts"]
    ):
        raise ValueError(f"{context}: valid sample did not generate the requested length")
    if not row["valid"] and not row.get("invalidReason"):
        raise ValueError(f"{context}: invalid sample needs invalidReason")


def pair_issues(rows, candidate, baseline):
    grouped = collections.defaultdict(dict)
    issues = []
    for row in rows:
        if row["engine"] not in (candidate, baseline):
            continue
        key = (row["workload"], row["warmup"], row["pairIndex"])
        if row["engine"] in grouped[key]:
            issues.append(f"duplicate {row['engine']} sample for {key}")
        grouped[key][row["engine"]] = row
    for key, pair in sorted(grouped.items()):
        if set(pair) != {candidate, baseline}:
            issues.append(f"incomplete candidate/baseline pair for {key}")
            continue
        left, right = pair[candidate], pair[baseline]
        for field in ("input", "output", "sampler"):
            if field == "output":
                left_value = left[field]["generatedTokenSHA256"]
                right_value = right[field]["generatedTokenSHA256"]
            else:
                left_value, right_value = left[field], right[field]
            if left_value != right_value:
                issues.append(f"{field} mismatch for {key}")
        if left["pairOrder"] != right["pairOrder"]:
            issues.append(f"pair order mismatch for {key}")
    for workload in sorted({row["workload"] for row in rows}):
        valid = [row for row in rows if row["workload"] == workload and row["valid"]]
        if valid:
            input_hashes = {tuple(row["input"]["renderedTokenSHA256"]) for row in valid}
            output_hashes = {tuple(row["output"]["generatedTokenSHA256"]) for row in valid}
            if len(input_hashes) != 1:
                issues.append(f"rendered input changed across trials for {workload}")
            if len(output_hashes) != 1:
                issues.append(f"greedy output changed across engines or trials for {workload}")
    return issues


def unavailable_metrics(rows, required_metrics):
    locations = {
        "ttftMilliseconds": "timing",
        "prefillTokensPerSecond": "timing",
        "decodeTokensPerSecond": "timing",
        "endToEndMilliseconds": "timing",
        "aggregateTokensPerSecond": "timing",
        "peakResidentBytes": "memory",
        "steadyResidentBytes": "memory",
        "peakMLXBytes": "memory",
        "cachedMLXBytes": "memory",
        "targetForwardCount": "speculation",
        "draftProposedTokens": "speculation",
        "draftAcceptedTokens": "speculation",
        "draftRounds": "speculation",
    }
    missing = []
    for metric in required_metrics:
        location = locations[metric]
        if any(row[location][metric] is None for row in rows):
            missing.append(metric)
    return missing


def summarize(protocol, session, rows, samples_digest):
    comparison = protocol["comparison"]
    candidate = comparison["candidateEngine"]
    baseline = comparison["baselineEngine"]
    measured_count = protocol["measurement"]["measuredTrialsPerCell"]
    warmup_count = protocol["measurement"]["warmupTrialsPerCell"]
    maximum_relative_mad = protocol["measurement"]["noisePolicy"]["maximumRelativeMAD"]
    maximum_drift = protocol["measurement"]["noisePolicy"]["maximumFirstToLastMedianDrift"]
    workloads = {workload["id"]: workload for workload in protocol["workloads"]}
    cells = []
    complete = True

    for workload_id, workload in workloads.items():
        engines = {}
        cell_complete = True
        cell_noise = []
        for engine in (candidate, baseline, *comparison["ordinaryAblations"]):
            all_rows = [row for row in rows if row["workload"] == workload_id and row["engine"] == engine]
            warmups = [row for row in all_rows if row["warmup"] and row["valid"]]
            measured = [row for row in all_rows if not row["warmup"] and row["valid"]]
            measured.sort(key=lambda row: row["trial"])
            if len(warmups) < warmup_count or len(measured) < measured_count:
                cell_complete = False

            metrics = {}
            for owner in ("timing", "memory", "speculation"):
                for metric in measured[0][owner] if measured else []:
                    if metric == "fallbacks":
                        continue
                    values = []
                    for row in measured:
                        value = row[owner][metric]
                        if isinstance(value, list):
                            values.extend(value)
                        elif value is not None:
                            values.append(value)
                    metrics[metric] = describe(values)

            primary = "aggregateTokensPerSecond" if workload["requests"] > 1 else "decodeTokensPerSecond"
            primary_values = [row["timing"][primary] for row in measured if row["timing"][primary] is not None]
            primary_stats = metrics.get(primary)
            relative_drift = drift(primary_values)
            if primary_stats is None:
                cell_complete = False
            else:
                if primary_stats["relativeMAD"] is not None and primary_stats["relativeMAD"] > maximum_relative_mad:
                    cell_noise.append(f"{engine} relative MAD exceeds limit")
                if relative_drift is not None and relative_drift > maximum_drift:
                    cell_noise.append(f"{engine} first/last drift exceeds limit")
            engines[engine] = {
                "warmupSamples": len(warmups),
                "measuredSamples": len(measured),
                "invalidSamples": len([row for row in all_rows if not row["valid"]]),
                "unavailableMetrics": unavailable_metrics(measured, protocol["requiredMetrics"]),
                "primaryMetric": primary,
                "primaryRelativeDrift": relative_drift,
                "metrics": metrics,
            }
            if engines[engine]["unavailableMetrics"]:
                cell_complete = False

        candidate_primary = engines[candidate]["metrics"].get(engines[candidate]["primaryMetric"])
        baseline_primary = engines[baseline]["metrics"].get(engines[baseline]["primaryMetric"])
        candidate_ttft = engines[candidate]["metrics"].get("ttftMilliseconds")
        baseline_ttft = engines[baseline]["metrics"].get("ttftMilliseconds")
        candidate_e2e = engines[candidate]["metrics"].get("endToEndMilliseconds")
        baseline_e2e = engines[baseline]["metrics"].get("endToEndMilliseconds")
        ratios = None
        outcome = "inconclusive"
        if all((candidate_primary, baseline_primary, candidate_ttft, baseline_ttft, candidate_e2e, baseline_e2e)):
            ratios = {
                "throughput": candidate_primary["median"] / baseline_primary["median"],
                "ttft": candidate_ttft["median"] / baseline_ttft["median"],
                "endToEnd": candidate_e2e["median"] / baseline_e2e["median"],
            }
            latency_match = (
                ratios["ttft"] <= comparison["maximumLatencyRatioForMatch"]
                and ratios["endToEnd"] <= comparison["maximumLatencyRatioForMatch"]
            )
            if cell_complete and not cell_noise and latency_match:
                if ratios["throughput"] >= comparison["minimumThroughputRatioForExceed"]:
                    outcome = "exceed"
                elif ratios["throughput"] >= comparison["minimumThroughputRatioForMatch"]:
                    outcome = "match"
                else:
                    outcome = "below"
            elif cell_complete and not cell_noise:
                outcome = "below"

        complete = complete and cell_complete and not cell_noise
        cells.append(
            {
                "workload": workload_id,
                "complete": cell_complete,
                "noiseIssues": cell_noise,
                "engines": engines,
                "candidateToBaselineRatios": ratios,
                "outcome": outcome,
            }
        )

    issues = pair_issues(rows, candidate, baseline)
    complete = complete and not issues and all(cell["outcome"] in ("match", "exceed", "below") for cell in cells)
    objective_met = complete and all(cell["outcome"] in ("match", "exceed") for cell in cells)
    return {
        "schemaVersion": 1,
        "protocol": protocol,
        "session": session,
        "generatedAtUTC": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rawSamplesSHA256": samples_digest,
        "rawSampleCount": len(rows),
        "pairIssues": issues,
        "complete": complete,
        "objectiveMet": objective_met,
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("protocol")
    parser.add_argument("session")
    parser.add_argument("samples")
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()

    protocol = load_json(arguments.protocol)
    session = load_json(arguments.session)
    rows, samples_digest = load_json_lines(arguments.samples)
    require(protocol, ["schemaVersion", "protocolId", "comparison", "sampling", "measurement", "workloads", "requiredMetrics"], "protocol")
    if protocol["schemaVersion"] != 1:
        raise ValueError("unsupported protocol schema")
    engine_ids = validate_session(protocol, session)
    workloads = {workload["id"]: workload for workload in protocol["workloads"]}
    for index, row in enumerate(rows, 1):
        validate_sample(row, index, protocol, session, engine_ids, workloads)
    ledger = summarize(protocol, session, rows, samples_digest)
    json.dump(ledger, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    if arguments.require_complete and not ledger["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(str(error)) from error
