"""Offline, paired AI evaluation release gates. Python standard library only."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys

MAX_BYTES = 8_000_000
MAX_CASES = 5_000


class InvalidRun(ValueError):
    """The comparison would not be trustworthy with these inputs."""


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidRun(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def number(value, name, low=0, high=1e12):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise InvalidRun(f"invalid {name}: expected finite number in [{low}, {high}]")
    return value


def label(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]{1,100}", value):
        raise InvalidRun(f"invalid {name}")
    return value


def load_run(path):
    """Reject partial, ambiguous or mismatched runs instead of dropping bad rows."""
    with open(path, "rb") as handle:
        raw = handle.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise InvalidRun("input exceeds 8 MB")
    rows = {}
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line, object_pairs_hook=strict_object)
        fields = {"id", "case_hash", "suite", "scorer", "run", "tags", "score", "latency_ms"}
        if not isinstance(row, dict) or set(row) != fields:
            raise InvalidRun(f"line {line_no}: incorrect fields")
        for field in ("id", "suite", "scorer", "run"):
            label(row[field], field)
        if not isinstance(row["case_hash"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["case_hash"]):
            raise InvalidRun("case_hash must be lowercase SHA-256")
        tags = row["tags"]
        if not isinstance(tags, list) or len(tags) > 8:
            raise InvalidRun("tags must be a list of at most 8 labels")
        for tag in tags:
            label(tag, "tag")
        if len(set(tags)) != len(tags):
            raise InvalidRun("duplicate tags")
        number(row["score"], "score", 0, 1)
        number(row["latency_ms"], "latency_ms")
        if row["id"] in rows:
            raise InvalidRun(f"duplicate case: {row['id']}")
        rows[row["id"]] = {**row, "tags": sorted(tags)}
        if len(rows) > MAX_CASES:
            raise InvalidRun("too many cases")
    if not rows:
        raise InvalidRun("empty run")
    for field in ("suite", "scorer", "run"):
        if len({r[field] for r in rows.values()}) != 1:
            raise InvalidRun(f"mixed {field} identifiers")
    return rows


def quantile(values, probability):
    """Linear interpolation at (n-1)*p, including endpoints."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    left = math.floor(position)
    right = math.ceil(position)
    return ordered[left] + (ordered[right] - ordered[left]) * (position - left)


def bootstrap(deltas, samples, seed):
    rng = random.Random(seed)
    n = len(deltas)
    means = [statistics.fmean(rng.choices(deltas, k=n)) for _ in range(samples)]
    return [quantile(means, 0.025), quantile(means, 0.975)]


def fingerprint(rows):
    canonical = json.dumps([rows[k] for k in sorted(rows)], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def compare(baseline_path, candidate_path, *, margin=0.02, min_cases=20,
            samples=2000, seed=42, max_p95_ms=1000):
    number(margin, "margin", 0, 1)
    number(max_p95_ms, "max_p95_ms")
    for value, name, low, high in ((min_cases, "min_cases", 2, MAX_CASES),
                                   (samples, "samples", 100, 10000),
                                   (seed, "seed", 0, 2**32 - 1)):
        if type(value) is not int or not low <= value <= high:
            raise InvalidRun(f"invalid {name}")
    base, candidate = load_run(baseline_path), load_run(candidate_path)
    if base.keys() != candidate.keys():
        raise InvalidRun("case IDs differ; missing or extra cases cannot be ignored")
    ids = sorted(base)
    for case in ids:
        for field in ("case_hash", "suite", "scorer", "tags"):
            if base[case][field] != candidate[case][field]:
                raise InvalidRun(f"case {case}: {field} differs")
    tags = sorted({t for row in base.values() for t in row["tags"]})
    if len(tags) > 32:
        raise InvalidRun("at most 32 tag slices are supported")
    groups = [("overall", ids)] + [("tag:" + tag, [k for k in ids if tag in base[k]["tags"]]) for tag in tags]
    # Bound work before generating bootstrap draws. No silent downsampling.
    if samples * sum(len(members) for _, members in groups) > 30_000_000:
        raise InvalidRun("bootstrap work limit exceeded; use fewer samples or tags")
    results = []
    for name, members in groups:
        deltas = [candidate[k]["score"] - base[k]["score"] for k in members]
        interval = None
        status = "insufficient_data"
        if len(members) >= min_cases:
            # A per-slice seed means unrelated new tags do not perturb this interval.
            group_seed = int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest(), "big")
            interval = bootstrap(deltas, samples, group_seed)
            if interval[0] >= -margin:
                status = "pass"
            elif interval[1] < -margin:
                status = "regression"
            else:
                status = "inconclusive"
        p95 = quantile([candidate[k]["latency_ms"] for k in members], 0.95)
        results.append({"slice": name, "n": len(members), "mean_delta": statistics.fmean(deltas),
                        "baseline_mean": statistics.fmean(base[k]["score"] for k in members),
                        "candidate_mean": statistics.fmean(candidate[k]["score"] for k in members),
                        "interval_95": interval, "quality_status": status,
                        "candidate_p95_ms": p95, "latency_status": "pass" if p95 <= max_p95_ms else "over_budget"})
    blockers = [{"slice": r["slice"], "quality": r["quality_status"], "latency": r["latency_status"]}
                for r in results if r["quality_status"] != "pass" or r["latency_status"] != "pass"]
    worst = sorted(ids, key=lambda k: (candidate[k]["score"] - base[k]["score"], k))
    return {"version": 1, "decision": "block" if blockers else "pass",
            "baseline_run": base[ids[0]]["run"], "candidate_run": candidate[ids[0]]["run"],
            "suite": base[ids[0]]["suite"], "scorer": base[ids[0]]["scorer"],
            "baseline_sha256": fingerprint(base), "candidate_sha256": fingerprint(candidate),
            "config": {"margin": margin, "min_cases": min_cases, "samples": samples,
                       "seed": seed, "max_p95_ms": max_p95_ms},
            "method": "paired percentile bootstrap; unadjusted per-slice 95% intervals",
            "blockers": blockers, "slices": results,
            "regressions": [{"id": k, "delta": candidate[k]["score"] - base[k]["score"],
                             "tags": base[k]["tags"]} for k in worst if candidate[k]["score"] < base[k]["score"]][:10]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--min-cases", type=int, default=20)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-p95-ms", type=float, default=1000)
    args = parser.parse_args(argv)
    try:
        report = compare(args.baseline, args.candidate, margin=args.margin,
                         min_cases=args.min_cases, samples=args.samples,
                         seed=args.seed, max_p95_ms=args.max_p95_ms)
    except (ValueError, OSError) as exc:
        print(f"Invalid comparison: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["decision"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
