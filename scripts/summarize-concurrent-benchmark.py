#!/usr/bin/env python3
"""Summarize FMLX_BENCHMARK JSON lines from an Xcode benchmark log."""

import collections
import json
import math
import statistics
import sys


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def main(path):
    groups = collections.defaultdict(list)
    with open(path) as source:
        for line in source:
            if "FMLX_BENCHMARK " not in line:
                continue
            row = json.loads(line.split("FMLX_BENCHMARK ", 1)[1])
            if row["trial"] == 0:
                continue
            groups[(row["activeLimit"], row["warm"], row["duringDecode"], row.get("batchDecode", True), row.get("backend", "scheduler"))].append(row)
    if not groups:
        raise SystemExit("No measured rounds found (trial zero is warmup).")
    for key, rows in sorted(groups.items()):
        result = dict(activeLimit=key[0], warm=key[1], duringDecode=key[2], batchDecode=key[3], backend=key[4], rounds=len(rows))
        for workload in ("short", "conversation", "background"):
            measurements = [m for row in rows for m in row["measurements"] if m["workload"] == workload]
            latencies = [(m["firstToken"] - m["submitted"]) * 1000 for m in measurements]
            result[workload] = dict(
                ttftP50Ms=statistics.median(latencies), ttftP95Ms=percentile(latencies, .95),
                reusedPrefixTokens=sorted({m["reusedPrefixTokens"] for m in measurements}))
        throughputs = []
        for row in rows:
            measurements = row["measurements"]
            elapsed = max(m["finished"] for m in measurements) - min(m["submitted"] for m in measurements)
            throughputs.append(sum(m["tokens"] for m in measurements) / elapsed)
        result["aggregateTokensPerSecondP50"] = statistics.median(throughputs)
        result["peakMLXBytes"] = max(row["peakMLXBytes"] for row in rows)
        result["cachedMLXBytes"] = max(row["cachedMLXBytes"] for row in rows)
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: summarize-concurrent-benchmark.py LOG")
    main(sys.argv[1])
