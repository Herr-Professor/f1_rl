#!/usr/bin/env python3
"""
Run a single-model ablation suite and write a manifest suitable for reporting.

Goal: isolate the effect of each lever:
- tools on/off
- deep_reasoning on/off
- multi_env on/off
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


ABLATIONS: Dict[str, Dict[str, Any]] = {
    "base_single_turn": {"use_tools": False, "multi_turn": False, "deep_reasoning": False, "multi_env": False},
    "deep_reasoning": {"use_tools": False, "multi_turn": False, "deep_reasoning": True, "multi_env": False},
    "tools": {"use_tools": True, "multi_turn": False, "deep_reasoning": False, "multi_env": False},
    "tools_deep": {"use_tools": True, "multi_turn": False, "deep_reasoning": True, "multi_env": False},
    "multi_env": {"use_tools": False, "multi_turn": False, "deep_reasoning": False, "multi_env": True},
    "multi_env_deep": {"use_tools": False, "multi_turn": False, "deep_reasoning": True, "multi_env": True},
    "multi_env_tools": {"use_tools": True, "multi_turn": False, "deep_reasoning": False, "multi_env": True},
    "multi_env_tools_deep": {"use_tools": True, "multi_turn": False, "deep_reasoning": True, "multi_env": True},
}


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


def _run_eval(
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-examples", type=int, default=150)
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--outputs-dir", default="outputs/evals")
    ap.add_argument("--prime-bin", default=os.environ.get("PRIME_BIN", "prime"))
    ap.add_argument("--out", default="reports/ablations_manifest.json")
    args = ap.parse_args()

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "generated_at": int(time.time()),
        "environment": args.environment,
        "model": args.model,
        "runs": [],
    }

    for name, env_args in ABLATIONS.items():
        res_dir = _run_eval(
            prime_bin=args.prime_bin,
            environment=args.environment,
            model=args.model,
            num_examples=args.num_examples,
            rollouts_per_example=args.rollouts,
            max_tokens=args.max_tokens,
            env_args=env_args,
            outputs_dir=outputs_dir,
        )
        manifest["runs"].append(
            {
                "ablation": name,
                "env_args": env_args,
                "num_examples": args.num_examples,
                "rollouts_per_example": args.rollouts,
                "max_tokens": args.max_tokens,
                "results_dir": str(res_dir) if res_dir else None,
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

