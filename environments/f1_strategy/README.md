# F1 Strategy Environment

Train and evaluate LLMs as an F1 race strategist using (1) historical OpenF1-derived scenarios and (2) deterministic stress tests.

## IDs

- **Hub ID**: `herr-professor/f1-strategy`
- **Local module**: `f1_strategy.py`

## What This Environment Is Optimizing

Each example is a multiple-choice strategy decision (A/B/C/D). The rubric rewards:

- picking the correct option (by a deterministic parser)
- giving structured reasoning that references key race signals
- optionally: tool use + deep reasoning evidence (numeric option-time estimates)

## Deterministic Verifier Spec (Read This First)

### 1) What counts as a “choice” (A/B/C/D)?

We parse the model output with `_extract_final_choice`:

- We search the last ~12 non-empty lines (from bottom) for an explicit decision line:
  - `Final: A` / `Decision - B` / `Answer: C` / `Choice: D`
  - If multiple appear, **the last one wins**.
- If none exist, we only accept a fallback if the **last non-empty line** begins with `A`/`B`/`C`/`D` (e.g. `C) Stay out ...`).
- We explicitly **do not** treat time-estimate lines like `A: 120.3` as a decision.

### 2) How `correct_strategy` is defined

`correct_strategy = 1.0` iff the parsed final choice exactly matches the example’s `answer` (case-insensitive). Otherwise `0.0`.

### 3) Missing/ambiguous decisions

- If no decision is parsed, `correct_strategy = 0.0`.
- `final_choice_present` applies a deterministic penalty: `-0.3` when missing.

### 4) Label source and tie-breaks

There are two label regimes:

- **Historical label (OpenF1 dataset, non-deep mode)**: `answer` is derived by `scripts/build_openf1_dataset.py` using race-phase heuristics + simple outcome proxies (stored in `info`, e.g. `outcome_score`). This is intentionally conservative and “policy-like”, not oracle-optimal.
- **Strategy-model label (deep reasoning mode)**: when `deep_reasoning=True`, we recompute `answer` on load by minimizing the environment’s internal strategy model over options A/B/C/D (expected-time over a short horizon). This makes “deep reasoning mode” self-consistent and deterministic.
  - Override can be disabled per-example by setting `info.lock_answer=true` (used for stress tests).

## Dataset + Reproducibility

- **OpenF1 cache**: `data/openf1_scenarios.jsonl`
- **Stress tests**: `data/stress_scenarios.jsonl` (hand-authored, deterministic)
- Every example’s `info` includes:
  - `dataset_sha256` (hash of the JSONL file that was loaded)
  - `dataset_variant` (`openf1` or `stress`)

To hard-pin drift in experiments:

- pass `expected_dataset_sha256=<sha256>` to `load_environment(...)`
- pass `seed=<int>` when subselecting `num_examples` to make selection deterministic

## Modes (Environment Args)

| Arg | Default | Meaning |
|---|---:|---|
| `dataset_variant` | `"openf1"` | `"openf1"` (historical) or `"stress"` (adversarial) |
| `dataset_path` | `None` | Optional override path to a JSONL dataset |
| `expected_dataset_sha256` | `None` | Optional guardrail against dataset drift |
| `seed` | `None` | Deterministic subsampling when `num_examples > 0` |
| `eval_season` | `2024` | Season held out for eval (set `None` to disable split) |
| `eval_tracks` | `None` | Tracks held out for eval |
| `use_tools` | `False` | Enable tool environment and tool-use reward |
| `multi_turn` | `False` | Enable pit-wall follow-up environment |
| `deep_reasoning` | `True` | Add strategy-model block + numeric scoring + label recomputation |
| `multi_env` | `False` | Route examples to per-track envs via an EnvGroup |
| `max_tracks` | `4` | Track count cap for `multi_env` |
| `max_tokens` | `900` | Generation cap |

## Tools

When `use_tools=True`, the environment exposes 3 tools and tracks tool call count:

- `tire_deg_estimator(info)` -> textual severity
- `pit_delta_lookup(info)` -> pit loss + undercut window
- `weather_confidence(info)` -> coarse wet/dry assessment

The rubric includes `uses_tools` which:

- rewards `+0.1` if at least one tool call happened
- penalizes `-0.2` if tools are enabled but none were called

## Rubric (Weights + Outputs)

Rubrics are **additive**: total reward is `sum(metric_value * metric_weight)`.

Non-deep mode (default when `deep_reasoning=False`):

| metric | value range | weight | intent |
|---|---:|---:|---|
| `correct_strategy` | {0, 1} | 1.0 | must be correct |
| `final_choice_present` | {0, -0.3} | 1.0 | enforce a decision |
| `has_reasoning` | {0, 0.2} | 0.2 | non-trivial explanation |
| `mentions_key_factors` | [0, 0.35] | 0.35 | uses signals (tires/weather/gaps/pit/SC/traffic) |
| `acknowledges_uncertainty` | {0, 0.1} | 0.1 | tradeoffs/contingencies |
| `outcome_aligned` | {0, 0.05, 0.15} | 0.15 | small bonus if correct + good outcome proxy |
| `uses_tools` | {0.1, -0.2, 0} | 0.1 | tool discipline (only when enabled) |

Deep reasoning mode adds:

| metric | value range | weight | intent |
|---|---:|---:|---|
| `option_times_present` | {0, 0.1, 0.3} | 0.3 | prints A/B/C/D time estimates |
| `option_time_accuracy` | [0, 0.6] | 0.6 | estimates match strategy-model ground truth |

## “Obviously Elite” Evaluation Protocol

This repo includes scripts to produce baselines, CIs, and ablations without handwaving.

### Baselines (models x modes, with 95% CIs)

1) Run a baseline matrix:

```bash
cd environments/f1_strategy
python3 scripts/run_baselines.py \
  --env herr-professor/f1-strategy \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --model Qwen/Qwen3-4B-Thinking-2507 \
  --num-examples 150 --rollouts 4
```

2) Render a markdown table with bootstrap CIs:

```bash
python3 scripts/render_baselines.py --manifest reports/baselines_manifest.json --out reports/baselines.md
```

### Ablations (same model, levers on/off)

```bash
cd environments/f1_strategy
python3 scripts/run_ablations.py --model Qwen/Qwen3-4B-Instruct-2507 --num-examples 150 --rollouts 4
python3 scripts/render_ablations.py --manifest reports/ablations_manifest.json --out reports/ablations.md
```

### Stress Tests (adversarial, deterministic)

```bash
prime eval run herr-professor/f1-strategy \
  -m Qwen/Qwen3-4B-Instruct-2507 \
  -n 50 -r 2 -s --skip-upload \
  -a '{"dataset_variant":"stress","deep_reasoning":false,"use_tools":false}'
```

## Regenerating the OpenF1 Dataset

```bash
cd environments/f1_strategy
uv run python scripts/build_openf1_dataset.py --years 2023 2024 --max-sessions 20 --max-scenarios 400
```
