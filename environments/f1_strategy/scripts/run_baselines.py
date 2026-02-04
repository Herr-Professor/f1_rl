#!/usr/bin/env python3
"""
Run a baseline matrix via `prime eval run` and record where results landed.

This is designed for reproducible reporting:
- fixed env args per "mode"
- fixed n/r
- results saved locally (no upload) so we can compute CIs consistently

Example:
  python scripts/run_baselines.py \\
    --env herr-professor/f1-strategy \\
    --model Qwen/Qwen3-4B-Instruct-2507 \\
    --model Qwen/Qwen3-4B-Thinking-2507 \\
    --num-examples 50 --rollouts 4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_MODES: Dict[str, Dict[str, Any]] = {
    # Intentionally minimal; add more once you have clean baselines.
    "single_turn": {"use_tools": False, "multi_turn": False, "deep_reasoning": False, "multi_env": False},
    "deep_reasoning": {"use_tools": False, "multi_turn": False, "deep_reasoning": True, "multi_env": False},
    # Tools uses the StatefulToolEnv (multi-turn with tools regardless of multi_turn flag).
    "tools": {"use_tools": True, "multi_turn": False, "deep_reasoning": False, "multi_env": False},
    "tools_deep": {"use_tools": True, "multi_turn": False, "deep_reasoning": True, "multi_env": False},
}


@dataclass
class RunSpec:
    environment: str
    model: str
    mode: str
    num_examples: int
    rollouts_per_example: int
    max_tokens: Optional[int]
    env_args: Dict[str, Any]
    results_dir: Optional[str] = None


def _newest_results_dir(outputs_dir: Path, *, since: float) -> Optional[Path]:
    candidates = []
    for p in outputs_dir.rglob("results.jsonl"):
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if mtime >= since:
            candidates.append((mtime, p.parent))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _run_prime_eval(
    *,
    prime_bin: str,
    environment: str,
    model: str,
    num_examples: int,
    rollouts_per_example: int,
    max_tokens: Optional[int],
    env_args: Dict[str, Any],
    outputs_dir: Path,
) -> Optional[Path]:
    start = time.time()
    cmd = [
        prime_bin,
        "eval",
        "run",
        environment,
        "-m",
        model,
        "-n",
        str(num_examples),
        "-r",
        str(rollouts_per_example),
        "--save-results",
        "--skip-upload",
        "-a",
        json.dumps(env_args),
    ]
    if max_tokens is not None:
        cmd += ["-t", str(max_tokens)]

    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return _newest_results_dir(outputs_dir, since=start - 2.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", dest="environment", default="herr-professor/f1-strategy")
    ap.add_argument("--model", action="append", default=[], help="Repeatable model id (prime inference model id).")
    ap.add_argument("--mode", action="append", default=list(DEFAULT_MODES.keys()))
    ap.add_argument("--num-examples", type=int, default=50)
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--outputs-dir", default="outputs/evals")
    ap.add_argument("--prime-bin", default=os.environ.get("PRIME_BIN", "prime"))
    ap.add_argument("--out", default="reports/baselines_manifest.json")
    args = ap.parse_args()

    models: List[str] = args.model
    if not models:
        raise SystemExit("Provide at least one --model.")

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    modes = args.mode
    for m in modes:
        if m not in DEFAULT_MODES:
            raise SystemExit(f"Unknown mode: {m}. Available: {sorted(DEFAULT_MODES)}")

    runs: List[RunSpec] = []
    for model in models:
        for mode in modes:
            env_args = dict(DEFAULT_MODES[mode])
            runs.append(
                RunSpec(
                    environment=args.environment,
                    model=model,
                    mode=mode,
                    num_examples=args.num_examples,
                    rollouts_per_example=args.rollouts,
                    max_tokens=args.max_tokens,
                    env_args=env_args,
                )
            )

    manifest: Dict[str, Any] = {
        "generated_at": int(time.time()),
        "environment": args.environment,
        "runs": [],
    }

    for run in runs:
        res_dir = _run_prime_eval(
            prime_bin=args.prime_bin,
            environment=run.environment,
            model=run.model,
            num_examples=run.num_examples,
            rollouts_per_example=run.rollouts_per_example,
            max_tokens=run.max_tokens,
            env_args=run.env_args,
            outputs_dir=outputs_dir,
        )
        run.results_dir = str(res_dir) if res_dir else None
        manifest["runs"].append(
            {
                "environment": run.environment,
                "model": run.model,
                "mode": run.mode,
                "num_examples": run.num_examples,
                "rollouts_per_example": run.rollouts_per_example,
                "max_tokens": run.max_tokens,
                "env_args": run.env_args,
                "results_dir": run.results_dir,
            }
        )

    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

