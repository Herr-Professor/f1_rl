#!/usr/bin/env python3
"""
Render a baselines markdown table (mean reward +/- bootstrap 95% CI)
from a manifest produced by scripts/run_baselines.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from summarize_eval import _bootstrap_ci, _mean, _read_jsonl  # type: ignore


def _summarize_results_jsonl(results_path: Path, *, seed: int, iters: int) -> Tuple[float, Tuple[float, float], int]:
    rows = _read_jsonl(str(results_path))
    rewards = [float(r["reward"]) for r in rows if "reward" in r]
    m = _mean(rewards)
    ci = _bootstrap_ci(rewards, iters=iters, seed=seed)
    return m, ci, len(rewards)


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="reports/baselines_manifest.json")
    ap.add_argument("--out", default="reports/baselines.md")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap-iters", type=int, default=2000)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs: List[Dict[str, Any]] = data.get("runs") or []
    if not runs:
        raise SystemExit(f"No runs in manifest: {manifest_path}")

    rows_md = []
    rows_md.append("| model | mode | n | reward_mean | ci95 | results |")
    rows_md.append("|---|---:|---:|---:|---:|---|")

    for run in runs:
        results_dir = run.get("results_dir")
        if not results_dir:
            rows_md.append(f"| `{run.get('model')}` | `{run.get('mode')}` | {run.get('num_examples')} | (missing) | (missing) | (missing) |")
            continue

        results_path = Path(results_dir) / "results.jsonl"
        if not results_path.exists():
            rows_md.append(f"| `{run.get('model')}` | `{run.get('mode')}` | {run.get('num_examples')} | (missing) | (missing) | `{results_dir}` |")
            continue

        mean, (lo, hi), n = _summarize_results_jsonl(
            results_path, seed=args.seed, iters=args.bootstrap_iters
        )
        rows_md.append(
            f"| `{run.get('model')}` | `{run.get('mode')}` | {n} | {_fmt(mean)} | [{_fmt(lo)}, {_fmt(hi)}] | `{results_dir}` |"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows_md) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

