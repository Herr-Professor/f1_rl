# F1 Strategy Environment

**A reinforcement learning environment for F1 race strategy decisions.**

## Overview
- **Environment ID**: `f1-strategy`
- **Description**: Train LLMs to make race strategy decisions using real OpenF1 data
- **Tags**: `reinforcement-learning`, `formula-1`, `strategy`, `multi-turn`, `tool-use`

## Features

| Feature | Description |
|---------|-------------|
| **Real Data** | 150+ scenarios from 2023-2024 OpenF1 API |
| **Tools** | Tire degradation, pit delta, weather confidence |
| **Multi-Turn** | Pit wall follow-up questions for clarification |
| **Multi-Env** | Track-specific training with 10 circuit priors |

## Datasets
- **Source**: [OpenF1 API](https://openf1.org)
- **Endpoints**: sessions, laps, stints, pit, weather, race_control
- **Features**: pace_delta, degradation_slope, undercut_window, pit_loss_est, sc_risk_proxy

## Task Types

### Single-Turn (default)
Standard strategy decision with reasoning.

### Deep Reasoning Mode (`deep_reasoning=True`)
Requires the model to compute expected total time for each option using a provided strategy model
and report numeric estimates before choosing the final strategy. This mode scores numerical
accuracy as evidence of deeper reasoning.

### Multi-Turn (`multi_turn=True`)
Pit wall asks follow-up questions about:
- Tire degradation impact
- Rain considerations  
- Undercut viability
- Safety car risk

### Tool Use (`use_tools=True`)
Three callable tools:
- `tire_deg_estimator`: Returns degradation severity (low/medium/high)
- `pit_delta_lookup`: Returns estimated pit loss and undercut window
- `weather_confidence`: Returns rain/track conditions

## Quickstart

```bash
# Run evaluation
prime eval run f1-strategy

# With custom config
prime eval run f1-strategy -m gpt-4.1-mini -n 50 -r 4
```

## Environment Arguments

| Arg | Type | Default | Description |
|-----|------|---------|-------------|
| `num_examples` | int | -1 | Limit dataset size (-1 for all) |
| `eval_season` | int | 2024 | Season for eval split |
| `eval_tracks` | list | None | Tracks for eval split |
| `use_tools` | bool | False | Enable tool calling |
| `multi_turn` | bool | False | Enable pit wall follow-ups |
| `deep_reasoning` | bool | True | Require expected-time calculations and score numeric accuracy |
| `multi_env` | bool | False | Train across multiple tracks |
| `max_tracks` | int | 4 | Max circuits for multi-env |
| `max_tokens` | int | 512 | Max response tokens |

## Rubric Metrics

| Metric | Weight | Description |
|--------|--------|-------------|
| `correct_strategy` | 1.0 | Matches expected answer (A/B/C/D) |
| `has_reasoning` | 0.2 | Response >60 chars with explanation |
| `mentions_key_factors` | 0.3 | References tires, rain, gaps, undercut, pit, SC |
| `acknowledges_uncertainty` | 0.1 | Mentions risk, tradeoffs, contingencies |

## Track Priors

| Track | Pit Loss | Overtake Difficulty | SC Risk |
|-------|----------|---------------------|---------|
| Monaco | 22.5s | 0.95 | 0.45 |
| Singapore | 26.0s | 0.85 | 0.55 |
| Budapest | 22.0s | 0.80 | 0.40 |
| Zandvoort | 21.0s | 0.80 | 0.35 |
| Suzuka | 22.0s | 0.55 | 0.30 |
| Monza | 20.0s | 0.35 | 0.20 |

## Regenerate Dataset

```bash
cd environments/f1_strategy
uv run python scripts/build_openf1_dataset.py --years 2023 2024 --max-sessions 20 --max-scenarios 400
```

## Training Config

Use `configs/lab/f1-strategy-profound.toml` for hosted training:
```toml
model = "Qwen/Qwen3-4B-Instruct-2507"
max_steps = 200

[[env]]
id = "herr-professor/f1-strategy"
args = { use_tools = true, multi_turn = true, multi_env = true }
```
