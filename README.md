# EvalDelta

**A better average can hide a worse AI product.** EvalDelta compares two scored AI evaluation runs, checks individual categories, and returns a machine-readable release decision.

[![Tests](https://github.com/brohum10/evaldelta/actions/workflows/tests.yml/badge.svg)](https://github.com/brohum10/evaldelta/actions/workflows/tests.yml)

Python 3.12+ · standard library only · offline · no API key

## See the failure an average misses

```sh
python demo.py
```

The demo uses **100 hand-authored synthetic cases**, not responses from an actual model:

| Slice | Cases | Baseline score | Candidate score | Quality decision |
| --- | ---: | ---: | ---: | --- |
| Overall | 100 | 0.70 | 0.80 | Pass |
| Long context | 20 | 0.70 | 0.40 | Regression |
| Short context | 80 | 0.70 | 0.90 | Pass |

The final decision is **block**, despite a 10-percentage-point overall improvement. These numbers illustrate the checker; they are not an AI accuracy benchmark. The demo runs in a temporary directory and leaves no dataset behind.

## What it does

- Matches baseline and candidate by case ID, not row order.
- Rejects missing cases, changed case hashes, different scorers, and changed category labels.
- Estimates paired score-difference intervals using seeded percentile bootstrap resampling.
- Checks the overall population and every supplied tag slice independently.
- Blocks on regressions, inconclusive intervals, insufficient samples, or candidate p95 latency over budget.
- Reports the ten largest individual score drops for investigation.
- Includes normalized content fingerprints and configuration in every JSON report.
- Exits with a CI-friendly status: `0` pass, `1` block, `2` invalid comparison.

## Architecture

```text
baseline.jsonl ─┐
               ├─ strict validation ─ ID/hash/scorer alignment ─ paired deltas
candidate.jsonl┘                                                  │
                                      overall + tag slices ───────┤
                                                                  ▼
                                            bootstrap + latency checks
                                                                  │
                                                       JSON report + exit code
```

The scorer and inference system stay outside this program. This separation makes the comparator usable with a local model, a hosted API, retrieval experiments, or human ratings without giving it credentials or raw prompts. It does not run or judge an LLM itself.

## Compare your own runs

Download this repository, open its folder, and use Python 3.12 or newer. Nothing needs to be installed with pip.

Produce one JSON object per line in each input file:

```json
{"id":"case-001","case_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","suite":"support-v1","scorer":"rubric-v2","run":"baseline-v1","tags":["long_context"],"score":0.7,"latency_ms":240}
```

That hash is a placeholder. In your exporter, hash a canonical representation of the case input **and expected answer**. The tool checks equality of supplied hashes but cannot verify the hidden case content. Use the same suite, scorer version, tags, and case hash for both runs. Only the run label, score, and latency should change. Include failed inferences with a score defined by your rubric rather than dropping them. Latency must be measured by your inference runner with consistent timing boundaries.

All eight fields are required; additional fields are rejected. Scores must be finite numbers between 0 and 1, with higher meaning better. Latency is a nonnegative finite millisecond value. Labels contain only ASCII letters, digits, `_ . : / -` and are at most 100 characters. Up to 8 distinct tags per case and 32 distinct tags per run are supported. Untagged cases still contribute to the overall result. Raw prompts, outputs, and secrets do not belong in these files.

```sh
python evaldelta.py baseline.jsonl candidate.jsonl \
  --margin 0.02 --min-cases 20 --samples 2000 \
  --seed 42 --max-p95-ms 1000 > report.json
```

Choose the score-loss margin and latency budget **before** looking at candidate results. In this example, a score loss of up to 0.02 is allowed. The margin is absolute score units, not relative percent. Every slice uses the same margin and minimum sample count.

### Decision rules

For each group, compute `candidate_score - baseline_score` per case. Resample those **paired differences**, preserving pairing, and calculate a mean for each resample. The 2.5th and 97.5th percentiles form an approximate 95% interval:

| Condition | Quality status |
| --- | --- |
| Fewer than `min_cases` observations | `insufficient_data` |
| Lower interval bound ≥ negative margin | `pass` |
| Upper interval bound < negative margin | `regression` |
| Otherwise | `inconclusive` |

A pass requires **every** group to pass quality and latency. Candidate latency p95 uses linear interpolation at `(n-1)*0.95`; it is a descriptive budget check, not a statistical latency comparison. An identical run can pass at a nonnegative margin. `pass` means the configured empirical checks passed, not proof that a model is safe or universally equivalent.

### Reproducibility and scope

Row order and tag order do not change the result. Sorted case IDs, a global seed, and a slice-specific seed make repeated runs reproducible on the tested Python runtimes. Hashes cover normalized records, including measured scores, latencies, and run labels; they detect changes, but are not signatures or proof of trustworthy scoring.

Inputs are bounded to 8 MB and 5,000 cases per file. Bootstrap work is capped at 30 million resampled observations across all slices. Oversized work is rejected rather than silently sampling fewer cases. Time is approximately O(B × sum of slice sizes), plus percentile sorting; memory includes the input records and B bootstrap means for one slice at a time.

## Tests

```sh
python -m unittest -v
```

The 34 tests cover hidden subgroup regressions, inconclusive and underpowered comparisons, latency boundaries, ID/hash/scorer mismatches, duplicate JSON keys, invalid scores, resource limits, deterministic ordering, known quantiles, bootstrap fixtures, and real CLI exit codes. GitHub Actions runs the suite and demo on Python 3.12 and 3.14.

## Important limitations

- The example is synthetic. No real model quality, production savings, or inference latency has been measured by this project.
- Percentile bootstrap intervals are approximate and can be unreliable for small, skewed, or degenerate samples. Twenty cases is a configurable operational floor, not a power analysis.
- Intervals are **unadjusted per slice**, not simultaneous confidence guarantees. Overlapping tags and repeated experiments complicate interpretation. This is a conservative workflow gate, not a formal family-wise statistical test.
- Resampling assumes independent cases. Clustered conversations, repeated users, or multiple generations per prompt require an appropriate clustered/hierarchical method not implemented here.
- One scalar score hides rubric dimensions. Run separate comparisons for genuinely different metrics. Scorer quality and representative evaluation design remain your responsibility.
- There is no model adapter, dashboard, cloud integration, authentication, BCa correction, or automatic deployment. A release job must actually honor the process exit code.
- Treat evaluator data as untrusted. Do not upload private production logs; `.jsonl` files are ignored by default.

Method reference: [SciPy's bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) explains paired resampling and percentile intervals, and discusses alternative interval methods. EvalDelta implements its own small standard-library version; it does not depend on SciPy.

## License

MIT. See [LICENSE](LICENSE).
