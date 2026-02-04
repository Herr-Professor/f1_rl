#!/usr/bin/env python3
"""
Summarize Prime/Verifiers eval outputs (results.jsonl) into means + bootstrap CIs.

Typical usage:
  python scripts/summarize_eval.py --results outputs/evals/<run>/results.jsonl
  python scripts/summarize_eval.py --outputs-dir outputs/evals --glob "**/results.jsonl"

This script is intentionally dependency-light (stdlib only).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def _bootstrap_ci(
    xs: List[float],
    *,
    iters: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(xs)
    means = []
    for _ in range(iters):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        means.append(_mean(sample))
    means.sort()
    lo = means[int((alpha / 2.0) * iters)]
    hi = means[int((1.0 - alpha / 2.0) * iters) - 1]
    return lo, hi


@dataclass(frozen=True)
class Summary:
    n: int
    reward_mean: float
    reward_std: float
    reward_ci95: Tuple[float, float]
    metric_means: Dict[str, float]


def summarize_results(rows: List[Dict[str, Any]], *, seed: int = 0) -> Summary:
    rewards: List[float] = []
    metric_sums: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}

    for r in rows:
        if "reward" in r:
            rewards.append(float(r["reward"]))
        metrics = r.get("metrics") or {}
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                try:
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
                    metric_counts[k] = metric_counts.get(k, 0) + 1
                except Exception:
                    continue

    metric_means = {k: metric_sums[k] / metric_counts[k] for k in metric_sums if metric_counts.get(k)}
    return Summary(
        n=len(rewards),
        reward_mean=_mean(rewards),
        reward_std=_std(rewards),
        reward_ci95=_bootstrap_ci(rewards, seed=seed),
        metric_means=dict(sorted(metric_means.items(), key=lambda kv: kv[0])),
    )


def _format_float(x: float) -> str:
    if x != x:  # NaN
        return "nan"
    return f"{x:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", action="append", default=[], help="Path to results.jsonl (repeatable).")
    ap.add_argument("--outputs-dir", default=None, help="Directory to search for results.jsonl.")
    ap.add_argument("--glob", dest="glob_pat", default="**/results.jsonl", help="Glob under outputs-dir.")
    ap.add_argument("--bootstrap-iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", dest="json_out", default=None, help="Write JSON summary to this path.")
    args = ap.parse_args()

    paths: List[str] = list(args.results)
    if args.outputs_dir:
        paths.extend(glob.glob(os.path.join(args.outputs_dir, args.glob_pat), recursive=True))
    paths = [p for p in paths if os.path.isfile(p)]
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("No results.jsonl files found.")

    all_summaries: Dict[str, Any] = {}
    for p in paths:
        rows = _read_jsonl(p)
        summary = summarize_results(rows, seed=args.seed)
        # Recompute CI with requested iters.
        rewards = [float(r["reward"]) for r in rows if "reward" in r]
        ci95 = _bootstrap_ci(rewards, iters=args.bootstrap_iters, seed=args.seed)
        summary = Summary(
            n=summary.n,
            reward_mean=summary.reward_mean,
            reward_std=summary.reward_std,
            reward_ci95=ci95,
            metric_means=summary.metric_means,
        )

        name = os.path.relpath(p, start=os.getcwd())
        all_summaries[name] = {
            "n": summary.n,
            "reward_mean": summary.reward_mean,
            "reward_std": summary.reward_std,
            "reward_ci95": list(summary.reward_ci95),
            "metric_means": summary.metric_means,
        }

        lo, hi = summary.reward_ci95
        print(f"{name}")
        print(f"  n={summary.n}")
        print(f"  reward mean={_format_float(summary.reward_mean)} std={_format_float(summary.reward_std)} ci95=[{_format_float(lo)}, {_format_float(hi)}]")
        if summary.metric_means:
            top = sorted(summary.metric_means.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
            print("  top metrics (abs mean): " + ", ".join(f"{k}={_format_float(v)}" for k, v in top))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, sort_keys=True)
        print(f"Wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

