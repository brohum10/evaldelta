"""Synthetic regression fixture; no model calls or real performance claims."""

import hashlib
import json
from pathlib import Path
import tempfile

from evaldelta import compare


def fixture():
    baseline, candidate = [], []
    for index in range(100):
        tag = "long_context" if index < 20 else "short_context"
        row = {"id": f"case-{index:03}", "case_hash": hashlib.sha256(f"synthetic-case-{index}".encode()).hexdigest(),
               "suite": "synthetic-support-v1", "scorer": "hand-authored-demo-v1",
               "run": "baseline", "tags": [tag], "score": 0.7, "latency_ms": 200}
        baseline.append(row)
        candidate.append({**row, "tags": list(row["tags"]), "run": "candidate", "score": 0.4 if index < 20 else 0.9,
                          "latency_ms": 240})
    return baseline, candidate


def write_rows(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def main():
    baseline, candidate = fixture()
    with tempfile.TemporaryDirectory() as directory:
        base, cand = Path(directory) / "baseline.jsonl", Path(directory) / "candidate.jsonl"
        write_rows(base, baseline)
        write_rows(cand, candidate)
        report = compare(base, cand)
    assert report["decision"] == "block"
    assert report["slices"][0]["quality_status"] == "pass"
    print("SYNTHETIC DEMO — no model inference performed")
    print("Overall score improves from 0.70 to 0.80, but long-context quality drops to 0.40.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
