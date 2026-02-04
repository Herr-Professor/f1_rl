#!/usr/bin/env python3
"""
Render an ablations markdown table from reports/ablations_manifest.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from summarize_eval import _bootstrap_ci, _mean, _read_jsonl  # type: ignore


def _summarize(results_path: Path, *, seed: int, iters: int) -> Tuple[float, Tuple[float, float], int]:
    rows = _read_jsonl(str(results_path))
    rewards = [float(r["reward"]) for r in rows if "reward" in r]
    return _mean(rewards), _bootstrap_ci(rewards, iters=iters, seed=seed), len(rewards)


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="reports/ablations_manifest.json")
    ap.add_argument("--out", default="reports/ablations.md")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap-iters", type=int, default=2000)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs: List[Dict[str, Any]] = data.get("runs") or []
    if not runs:
        raise SystemExit(f"No runs in manifest: {manifest_path}")

    rows_md = []
    rows_md.append("| ablation | n | reward_mean | ci95 | results |")
    rows_md.append("|---|---:|---:|---:|---|")

    for run in runs:
        res_dir = run.get("results_dir")
        ab = run.get("ablation")
        if not res_dir:
            rows_md.append(f"| `{ab}` | {run.get('num_examples')} | (missing) | (missing) | (missing) |")
            continue
        results_path = Path(res_dir) / "results.jsonl"
        if not results_path.exists():
            rows_md.append(f"| `{ab}` | {run.get('num_examples')} | (missing) | (missing) | `{res_dir}` |")
            continue
        mean, (lo, hi), n = _summarize(results_path, seed=args.seed, iters=args.bootstrap_iters)
        rows_md.append(f"| `{ab}` | {n} | {_fmt(mean)} | [{_fmt(lo)}, {_fmt(hi)}] | `{res_dir}` |")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows_md) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

