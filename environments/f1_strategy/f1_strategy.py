import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import verifiers as vf
from datasets import Dataset


OPENF1_SCENARIOS_PATH = Path(__file__).resolve().parent / "data" / "openf1_scenarios.jsonl"

TRACK_PRIORS: Dict[str, Dict[str, float]] = {
    "Monaco": {"pit_loss": 22.5, "overtake_difficulty": 0.95, "sc_risk_base": 0.45},
    "Monza": {"pit_loss": 20.0, "overtake_difficulty": 0.35, "sc_risk_base": 0.20},
    "Singapore": {"pit_loss": 26.0, "overtake_difficulty": 0.85, "sc_risk_base": 0.55},
    "Spa": {"pit_loss": 23.0, "overtake_difficulty": 0.45, "sc_risk_base": 0.30},
    "Silverstone": {"pit_loss": 21.0, "overtake_difficulty": 0.40, "sc_risk_base": 0.25},
    "Suzuka": {"pit_loss": 22.0, "overtake_difficulty": 0.55, "sc_risk_base": 0.30},
    "Austin": {"pit_loss": 21.5, "overtake_difficulty": 0.45, "sc_risk_base": 0.30},
    "Bahrain": {"pit_loss": 20.5, "overtake_difficulty": 0.40, "sc_risk_base": 0.25},
    "Sakhir": {"pit_loss": 20.5, "overtake_difficulty": 0.40, "sc_risk_base": 0.25},  # Bahrain alias
    "Zandvoort": {"pit_loss": 21.0, "overtake_difficulty": 0.80, "sc_risk_base": 0.35},
    "Budapest": {"pit_loss": 22.0, "overtake_difficulty": 0.80, "sc_risk_base": 0.40},
    "Jeddah": {"pit_loss": 19.5, "overtake_difficulty": 0.50, "sc_risk_base": 0.60},  # High SC risk
    "Melbourne": {"pit_loss": 21.0, "overtake_difficulty": 0.70, "sc_risk_base": 0.45},
}

DEFAULT_TRACK_PROFILE = {"pit_loss": 22.0, "overtake_difficulty": 0.60, "sc_risk_base": 0.30}

COMPOUND_PARAMS: Dict[str, Dict[str, float]] = {
    # pace_offset in seconds (negative = faster), deg_factor multiplies slope
    # dry_penalty and wet_penalty are expected per-lap penalties based on rain probability
    "soft": {"pace_offset": -0.25, "deg_factor": 1.15, "dry_penalty": 0.0, "wet_penalty": 2.0},
    "medium": {"pace_offset": 0.00, "deg_factor": 1.00, "dry_penalty": 0.0, "wet_penalty": 2.0},
    "hard": {"pace_offset": 0.20, "deg_factor": 0.85, "dry_penalty": 0.0, "wet_penalty": 2.0},
    "intermediates": {"pace_offset": 0.80, "deg_factor": 1.05, "dry_penalty": 0.9, "wet_penalty": 0.3},
    "wet": {"pace_offset": 1.40, "deg_factor": 1.10, "dry_penalty": 1.6, "wet_penalty": 0.1},
}

DEFAULT_COMPOUND_PARAMS = {"pace_offset": 0.10, "deg_factor": 1.00, "dry_penalty": 0.0, "wet_penalty": 2.0}

STRATEGY_MODEL_DEFAULTS = {
    "horizon_laps": 6,
    "delay_laps": 3,
    "fresh_deg_factor": 0.60,
}


def _track_profile(track: str) -> Dict[str, float]:
    return TRACK_PRIORS.get(track, DEFAULT_TRACK_PROFILE)


def _compound_profile(compound: str) -> Dict[str, float]:
    if not compound:
        return DEFAULT_COMPOUND_PARAMS
    return COMPOUND_PARAMS.get(compound.lower(), DEFAULT_COMPOUND_PARAMS)


def _parse_info(info) -> Dict[str, Any]:
    if isinstance(info, dict):
        return info
    if isinstance(info, str):
        try:
            return json.loads(info)
        except json.JSONDecodeError:
            return {}
    return {}


def _system_prompt(use_tools: bool, multi_turn: bool) -> str:
    tool_line = (
        "You may use tools for tire degradation, pit delta, or weather confidence. "
        "Your first assistant message must be a tool call. "
        "After tools, provide a final line in the format: Final: <A/B/C/D>."
        if use_tools
        else "Use the provided signals; do not invent data."
    )
    follow_up = (
        "The pit wall may ask one follow-up question. Answer it concisely and restate your decision with a Final line."
        if multi_turn
        else "State your decision clearly as a single letter (A/B/C/D) with reasoning."
    )
    return (
        "You are an F1 race strategist. Optimize for expected points with disciplined risk management. "
        "Be explicit about tradeoffs, uncertainty, and second-order effects (track position vs pace). "
        f"{tool_line} {follow_up}"
    )


def _format_signal(value, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value}{suffix}"


def _format_bool(value: Optional[bool]) -> str:
    if value is None:
        return "Unknown"
    return "Yes" if value else "No"


def _extract_final_choice(text: str) -> Optional[str]:
    if not text:
        return None
    tail_lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    tail = "\n".join(tail_lines[-3:]) if tail_lines else text
    match = re.search(r"(?:^|\n)\s*(?:final|decision|answer|choice)\s*[:\\-]\\s*([ABCD])\\b", tail, re.I)
    if match:
        return match.group(1).upper()
    match = re.search(r"(?:^|\n)\s*([ABCD])\s*(?:$|[).])", tail)
    if match:
        return match.group(1).upper()
    return None


def _get_pit_compound_option(current_compound: str, has_rain: bool) -> Tuple[str, str]:
    """Return (option_a_compound, option_b_compound) based on current tires and weather."""
    current = current_compound.lower() if current_compound else "medium"
    if has_rain:
        return ("intermediates", "wet")
    # Offer a different compound than what we're on
    if current == "soft":
        return ("medium", "hard")
    elif current == "hard":
        return ("soft", "medium")
    else:  # medium
        return ("soft", "hard")


def _rain_probability(rainfall: float, rain_soon: bool) -> float:
    if rainfall >= 0.5:
        return 0.9
    if rainfall > 0.1:
        return 0.6
    if rain_soon:
        return 0.4
    return 0.1


def _compound_effects(compound: str, rain_prob: float) -> Dict[str, float]:
    profile = _compound_profile(compound)
    rain_penalty = profile["dry_penalty"] * (1 - rain_prob) + profile["wet_penalty"] * rain_prob
    return {
        "pace_offset": profile["pace_offset"],
        "deg_factor": profile["deg_factor"],
        "rain_penalty": rain_penalty,
    }


def _strategy_model_params(info: Dict[str, Any]) -> Dict[str, Any]:
    track = info.get("track", "Unknown")
    profile = _track_profile(track)
    rainfall = float(info.get("rainfall", 0.0) or 0.0)
    rain_soon = bool(info.get("rain_soon", False))
    rain_prob = _rain_probability(rainfall, rain_soon)
    current_compound = str(info.get("tire_compound", "medium")).lower()
    has_rain = rainfall > 0 or rain_soon
    pit_option_a, pit_option_b = _get_pit_compound_option(current_compound, has_rain)

    pit_loss = float(info.get("pit_loss_est") or profile["pit_loss"])
    sc_risk = info.get("sc_risk_proxy")
    sc_risk = float(sc_risk) if sc_risk is not None else profile["sc_risk_base"]
    pit_loss_now = pit_loss * (1.0 + 0.6 * sc_risk)
    pit_loss_later = max(pit_loss * 0.5, pit_loss * (1.0 - 0.6 * sc_risk))

    base_lap = float(info.get("lap_duration") or 90.0)
    undercut_window = info.get("undercut_window")
    if undercut_window is not None:
        fresh_gain = max(0.0, min(1.2, pit_loss - float(undercut_window)))
    else:
        fresh_gain = 0.6
    base_lap_new = max(0.0, base_lap - fresh_gain)

    deg_slope = float(info.get("degradation_slope") or 0.05)
    deg_slope_new = deg_slope * STRATEGY_MODEL_DEFAULTS["fresh_deg_factor"]

    traffic_gap = info.get("traffic_tightness")
    traffic_gap = float(traffic_gap) if traffic_gap is not None else 99.0
    overtake_diff = profile["overtake_difficulty"]
    traffic_penalty_now = max(0.0, 1.5 - traffic_gap) * (1.0 + 1.5 * overtake_diff)
    traffic_penalty_later = traffic_penalty_now * 0.5

    horizon_laps = STRATEGY_MODEL_DEFAULTS["horizon_laps"]
    delay_laps = STRATEGY_MODEL_DEFAULTS["delay_laps"]

    return {
        "track": track,
        "current_compound": current_compound,
        "pit_option_a": pit_option_a,
        "pit_option_b": pit_option_b,
        "has_rain": has_rain,
        "rain_prob": rain_prob,
        "pit_loss_now": pit_loss_now,
        "pit_loss_later": pit_loss_later,
        "traffic_penalty_now": traffic_penalty_now,
        "traffic_penalty_later": traffic_penalty_later,
        "base_lap": base_lap,
        "base_lap_new": base_lap_new,
        "deg_slope": deg_slope,
        "deg_slope_new": deg_slope_new,
        "horizon_laps": horizon_laps,
        "delay_laps": delay_laps,
        "compound_effects": {
            "current": _compound_effects(current_compound, rain_prob),
            "option_a": _compound_effects(pit_option_a, rain_prob),
            "option_b": _compound_effects(pit_option_b, rain_prob),
        },
    }


def _expected_stint_time(
    base_lap: float,
    deg_slope: float,
    compound: str,
    rain_prob: float,
    laps: int,
) -> float:
    if laps <= 0:
        return 0.0
    effects = _compound_effects(compound, rain_prob)
    per_lap_base = base_lap + effects["pace_offset"] + effects["rain_penalty"]
    deg_term = deg_slope * effects["deg_factor"]
    return per_lap_base * laps + deg_term * (laps * (laps + 1) / 2.0)


def _expected_option_times(info: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    params = _strategy_model_params(info)
    horizon = params["horizon_laps"]
    delay = min(params["delay_laps"], horizon)

    rain_prob = params["rain_prob"]
    base_lap = params["base_lap"]
    base_lap_new = params["base_lap_new"]
    deg_slope = params["deg_slope"]
    deg_slope_new = params["deg_slope_new"]

    current = params["current_compound"]
    opt_a = params["pit_option_a"]
    opt_b = params["pit_option_b"]

    pit_loss_now = params["pit_loss_now"]
    pit_loss_later = params["pit_loss_later"]
    traffic_now = params["traffic_penalty_now"]
    traffic_later = params["traffic_penalty_later"]

    time_a = pit_loss_now + traffic_now + _expected_stint_time(
        base_lap_new, deg_slope_new, opt_a, rain_prob, horizon
    )
    time_b = pit_loss_now + traffic_now + _expected_stint_time(
        base_lap_new, deg_slope_new, opt_b, rain_prob, horizon
    )
    time_c = _expected_stint_time(base_lap, deg_slope, current, rain_prob, horizon)

    time_d = (
        _expected_stint_time(base_lap, deg_slope, current, rain_prob, delay)
        + pit_loss_later
        + traffic_later
        + _expected_stint_time(base_lap_new, deg_slope_new, opt_a, rain_prob, horizon - delay)
    )

    return {"A": time_a, "B": time_b, "C": time_c, "D": time_d}, params


def _extract_option_times(text: str) -> Dict[str, float]:
    if not text:
        return {}
    pattern = re.compile(r"\b([ABCD])\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.I)
    matches = pattern.findall(text)
    results: Dict[str, float] = {}
    for key, value in matches:
        try:
            results[key.upper()] = float(value)
        except ValueError:
            continue
    return results


def _build_prompt_text(info: Dict[str, Any], use_tools: bool, multi_turn: bool, deep_reasoning: bool) -> str:
    track = info.get("track", "Unknown")
    profile = _track_profile(track)
    rainfall = info.get("rainfall", 0.0) or 0.0
    rain_soon = info.get("rain_soon", False)
    has_rain = rainfall > 0 or rain_soon
    weather_summary = "Rain at circuit" if rainfall > 0 else ("Rain expected soon" if rain_soon else "Clear, no changes expected")
    current_compound = str(info.get("tire_compound", "medium"))

    strategy_params = _strategy_model_params(info)
    pit_option_a = strategy_params["pit_option_a"]
    pit_option_b = strategy_params["pit_option_b"]

    signals = [
        f"PaceΔ: {_format_signal(info.get('pace_delta'), 's')} (last 3 vs stint median)",
        f"Deg Slope: {_format_signal(info.get('degradation_slope'), ' s/lap')}",
        f"Undercut: {_format_signal(info.get('undercut_window'), 's')}",
        f"Pit Loss: {_format_signal(info.get('pit_loss_est') or profile['pit_loss'], 's')}",
        f"SC Risk: {_format_signal(info.get('sc_risk_proxy'), '')} (base {profile['sc_risk_base']})",
        f"Traffic: {_format_signal(info.get('traffic_tightness'), 's')} (gap ahead)",
        f"DRS Train: {_format_bool(info.get('drs_train_proxy'))}",
        f"PosΔ(5): {_format_signal(info.get('position_delta_recent'), '')}",
        f"Warmup: {_format_signal(info.get('warmup_risk'), '')}",
        f"Conflicts: {_format_signal(info.get('conflict_score'), '')}",
    ]

    context = f"""Race Status - Lap {info.get('lap', '?')}/{info.get('total_laps', '?')}
Position: P{info.get('position', '?')}
Current Tires: {current_compound.capitalize()} ({info.get('tire_age', '?')} laps old)
Gap to Leader: +{info.get('gap_to_leader', '?')}s
Gap to Car Ahead: +{info.get('interval', '?')}s
Fuel: {info.get('fuel_remaining', '?')}% remaining
Weather: {weather_summary}
Track: {track}
"""
    context += "\nStrategic Signals\n" + "\n".join(f"- {s}" for s in signals)

    priors = (
        f"Track Priors: pit loss ~{profile['pit_loss']}s, overtake difficulty {profile['overtake_difficulty']}, "
        f"baseline SC risk {profile['sc_risk_base']}"
    )

    strategy_block = ""
    if deep_reasoning:
        effects = strategy_params["compound_effects"]
        strategy_block = f"""

Strategy Model (compute expected total time over the next {strategy_params['horizon_laps']} laps)
- Horizon: {strategy_params['horizon_laps']} laps
- Option D delays {strategy_params['delay_laps']} laps, then pits to Option A compound
- Base lap (current tires): {strategy_params['base_lap']:.3f}s
- Base lap (fresh tires): {strategy_params['base_lap_new']:.3f}s
- Degradation slope (current): {strategy_params['deg_slope']:.4f} s/lap
- Degradation slope (fresh): {strategy_params['deg_slope_new']:.4f} s/lap
- Pit loss now: {strategy_params['pit_loss_now']:.1f}s; pit loss later: {strategy_params['pit_loss_later']:.1f}s
- Traffic penalty now: {strategy_params['traffic_penalty_now']:.2f}s; later: {strategy_params['traffic_penalty_later']:.2f}s
- Rain probability: {strategy_params['rain_prob']:.2f}
Compound effects (pace_offset, deg_factor, rain_penalty):
- Current ({current_compound}): {effects['current']['pace_offset']:+.2f}, {effects['current']['deg_factor']:.2f}, {effects['current']['rain_penalty']:.2f}
- Option A ({pit_option_a}): {effects['option_a']['pace_offset']:+.2f}, {effects['option_a']['deg_factor']:.2f}, {effects['option_a']['rain_penalty']:.2f}
- Option B ({pit_option_b}): {effects['option_b']['pace_offset']:+.2f}, {effects['option_b']['deg_factor']:.2f}, {effects['option_b']['rain_penalty']:.2f}
Formula: total(L, compound) = sum_(i=1..L)[base + pace_offset + rain_penalty + deg_slope*deg_factor*i]
Use base/deg_slope for current tires; use base_lap_new/deg_slope_new after a pit."""

    # Dynamic options based on tire and weather state
    if deep_reasoning:
        if has_rain:
            decision = f"""

What is your strategy?
A) Pit now for {pit_option_a} (rain-ready)
B) Pit now for {pit_option_b} (full wet)
C) Stay out on current {current_compound} tires
D) Pit when rain intensifies (delay then pit to {pit_option_a})

Compute expected total time for each option and answer in this format:
A: <time>
B: <time>
C: <time>
D: <time>
Decision: <A/B/C/D>
Final: <A/B/C/D>"""
        else:
            decision = f"""

What is your strategy?
A) Pit now for {pit_option_a} tires
B) Pit now for {pit_option_b} tires
C) Stay out on current {current_compound} tires
D) Extend stint and pit later (delay then pit to {pit_option_a})

Compute expected total time for each option and answer in this format:
A: <time>
B: <time>
C: <time>
D: <time>
Decision: <A/B/C/D>
Final: <A/B/C/D>"""
    else:
        if has_rain:
            decision = f"""

What is your strategy?
A) Pit now for {pit_option_a} (rain-ready)
B) Pit now for {pit_option_b} (full wet)
C) Stay out on current {current_compound} tires
D) Pit when rain intensifies

Reply with your choice and reasoning."""
        else:
            decision = f"""

What is your strategy?
A) Pit now for {pit_option_a} tires
B) Pit now for {pit_option_b} tires
C) Stay out on current {current_compound} tires
D) Extend stint and pit later

Reply with your choice and reasoning."""

    if multi_turn:
        decision += " If you need clarification, answer the pit wall question when asked."

    return f"{context}\n\n{priors}{strategy_block}{decision}"


def _build_prompt_messages(info: Dict[str, Any], use_tools: bool, multi_turn: bool, deep_reasoning: bool) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt(use_tools, multi_turn)},
        {"role": "user", "content": _build_prompt_text(info, use_tools, multi_turn, deep_reasoning)},
    ]


def _prepare_dataset(dataset: Dataset, use_tools: bool, multi_turn: bool, deep_reasoning: bool) -> Dataset:
    def _map(row, idx):
        info = _parse_info(row.get("info"))
        info.setdefault("track", row.get("track"))
        info.setdefault("season", row.get("season"))
        info.setdefault("traffic_tightness", row.get("traffic_tightness"))
        info.setdefault("drs_train_proxy", row.get("drs_train_proxy"))
        info.setdefault("position_delta_recent", row.get("position_delta_recent"))
        info.setdefault("warmup_risk", row.get("warmup_risk"))
        info.setdefault("conflict_score", row.get("conflict_score"))
        if deep_reasoning:
            try:
                expected_times, _ = _expected_option_times(info)
                if expected_times:
                    best = min(expected_times.items(), key=lambda kv: kv[1])[0]
                    row["answer"] = best
            except Exception:
                pass
        row["info"] = json.dumps(info)
        row["track"] = info.get("track") or "Unknown"
        row["season"] = info.get("season")
        row["prompt"] = _build_prompt_messages(info, use_tools, multi_turn, deep_reasoning)
        row["example_id"] = int(idx)
        return row

    return dataset.map(_map, with_indices=True)


def _split_train_eval(dataset: Dataset, eval_season: Optional[int], eval_tracks: Optional[List[str]]):
    if eval_season is None and not eval_tracks:
        return dataset, None

    def is_eval(row) -> bool:
        season = row.get("season")
        track = row.get("track")
        season_match = eval_season is not None and season == eval_season
        track_match = bool(eval_tracks) and track in eval_tracks
        return season_match or track_match

    eval_dataset = dataset.filter(is_eval)
    train_dataset = dataset.filter(lambda row: not is_eval(row))
    return train_dataset, eval_dataset


def _load_openf1_scenarios():
    if not OPENF1_SCENARIOS_PATH.exists():
        return []

    scenarios = []
    with OPENF1_SCENARIOS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                scenarios.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return scenarios


def create_f1_dataset(use_tools: bool = False, multi_turn: bool = False, deep_reasoning: bool = False) -> Dataset:
    """Create F1 racing strategy scenarios from OpenF1 cache."""
    openf1_scenarios = _load_openf1_scenarios()
    if not openf1_scenarios:
        raise RuntimeError(
            "OpenF1 scenarios not found. Run scripts/build_openf1_dataset.py to generate data/openf1_scenarios.jsonl."
        )
    dataset = Dataset.from_list(openf1_scenarios)
    return _prepare_dataset(dataset, use_tools, multi_turn, deep_reasoning)


def _pit_wall_question(info: Dict[str, Any], response: str) -> str:
    """Generate pit wall follow-up question based on response context."""
    response = response.lower()
    if ("tire" not in response and "tyre" not in response) or info.get("degradation_slope"):
        return "Pit wall: How does tire degradation influence your call over the next 5 laps?"
    if info.get("rainfall", 0.0) > 0 and "rain" not in response:
        return "Pit wall: Rain is present. How does that change your choice and timing?"
    if info.get("undercut_window") is not None and "undercut" not in response:
        return "Pit wall: Do you expect the undercut to work here given pit loss?"
    if info.get("sc_risk_proxy") is not None and "safety car" not in response and "vsc" not in response:
        return "Pit wall: How does safety car risk change the optimal window?"
    return "Pit wall: Confirm your decision and your main risk tradeoff."


class F1PitWallEnv(vf.MultiTurnEnv):
    """Multi-turn environment with pit wall follow-up questions."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        state["pit_wall_asked"] = False
        return await super().setup_state(state, **kwargs)

    async def env_response(self, messages: vf.Messages, state: vf.State) -> vf.Messages:
        if state.get("pit_wall_asked"):
            return []
        info = _parse_info(state.get("info"))
        response = messages[-1]["content"] if messages else ""
        question = _pit_wall_question(info, response)
        state["pit_wall_asked"] = True
        return [{"role": "user", "content": question}]


class F1ToolEnv(vf.StatefulToolEnv):
    """Tool environment with tire, pit delta, and weather tools."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_tool(self.tire_deg_estimator, args_to_skip=["info"])
        self.add_tool(self.pit_delta_lookup, args_to_skip=["info"])
        self.add_tool(self.weather_confidence, args_to_skip=["info"])

    def update_tool_args(self, tool_name, tool_args, messages, state, **kwargs):
        state["tool_call_count"] = int(state.get("tool_call_count") or 0) + 1
        tool_args["info"] = state.get("info")
        return tool_args

    @staticmethod
    def tire_deg_estimator(info: str) -> str:
        """Estimate tire degradation severity based on current stint data."""
        data = _parse_info(info)
        deg = data.get("degradation_slope", 0.05)
        if deg < 0.03:
            return "Tire degradation: LOW - tires performing well, no urgent need to pit."
        if deg < 0.07:
            return "Tire degradation: MEDIUM - performance dropping, consider pit in next 5-10 laps."
        return "Tire degradation: HIGH - significant pace loss, pit soon recommended."

    @staticmethod
    def pit_delta_lookup(info: str) -> str:
        """Look up estimated pit stop time loss and undercut window for current track."""
        data = _parse_info(info)
        track = data.get("track", "Unknown")
        profile = _track_profile(track)
        pit_loss = data.get("pit_loss_est", profile["pit_loss"])
        undercut = data.get("undercut_window", 2.5)
        return f"Pit loss at {track}: ~{pit_loss:.1f}s. Undercut window: {undercut:.1f}s."

    @staticmethod
    def weather_confidence(info: str) -> str:
        """Get weather forecast confidence and track conditions."""
        data = _parse_info(info)
        rainfall = data.get("rainfall", 0.0)
        if rainfall > 0.5:
            return "Weather: RAIN - track wet, intermediates or wets required."
        if rainfall > 0.1:
            return "Weather: LIGHT RAIN - track damp, consider intermediates."
        return "Weather: DRY - no precipitation expected."


class F1EnvGroupRubric(vf.Rubric):
    """Rubric that routes scoring by track info, not task."""

    def __init__(self, env_map: Dict[str, vf.Environment], env_names: List[str]):
        super().__init__()
        self.env_map = env_map
        self.env_names = env_names

        all_names = set()
        for env in env_map.values():
            all_names.update(env.rubric._get_reward_func_names())
        self.all_reward_names = sorted(all_names)

    def _resolve_env(self, state: vf.State) -> Optional[vf.Environment]:
        info = _parse_info(state.get("info"))
        track = info.get("track")
        if track in self.env_map:
            return self.env_map[track]
        task = state.get("task")
        if task in self.env_map:
            return self.env_map[task]
        if self.env_names:
            return self.env_map.get(self.env_names[0])
        return None

    def _empty_metrics(self) -> Dict[str, float]:
        return {name: 0.0 for name in self.all_reward_names}

    async def score_rollout(self, state: vf.State, score_sem) -> None:
        env = self._resolve_env(state)
        if env is None:
            state["reward"] = 0.0
            state["metrics"] = self._empty_metrics()
            return
        await env.rubric.score_rollout(state, score_sem=score_sem)
        metrics = self._empty_metrics()
        for name, value in (state.get("metrics") or {}).items():
            if name in metrics:
                metrics[name] = value
        state["metrics"] = metrics

    async def score_group(self, states: List[vf.State], score_sem) -> None:
        if not states:
            return
        env = self._resolve_env(states[0])
        if env is None:
            for state in states:
                state["reward"] = 0.0
                state["metrics"] = self._empty_metrics()
                state["timing"]["scoring_ms"] = 0.0
            return
        await env.rubric.score_group(states, score_sem=score_sem)
        for state in states:
            metrics = self._empty_metrics()
            for name, value in (state.get("metrics") or {}).items():
                if name in metrics:
                    metrics[name] = value
            state["metrics"] = metrics


class F1EnvGroup(vf.EnvGroup):
    """EnvGroup that routes by track while keeping task stable for Prime-RL."""

    def __init__(self, env_id: str, **kwargs):
        super().__init__(env_id=env_id, **kwargs)
        self.env_id = env_id
        self.rubric = F1EnvGroupRubric(self.env_map, self.env_names)
        if self.dataset is not None:
            self.dataset = self.dataset.map(lambda row: {**row, "task": env_id})
        if self.eval_dataset is not None:
            self.eval_dataset = self.eval_dataset.map(lambda row: {**row, "task": env_id})

    async def rollout(self, input, client, model, sampling_args=None):
        info = _parse_info(input.get("info"))
        track = info.get("track")
        env = self.env_map.get(track) if track in self.env_map else None
        if env is None:
            env = self.envs[0]
        routed = dict(input)
        routed["task"] = self.env_id
        return await env.rollout(routed, client, model, sampling_args)


async def correct_strategy(completion, answer) -> float:
    """Reward for choosing the correct strategy."""
    if not completion:
        return 0.0
    response = completion[-1]["content"]
    choice = _extract_final_choice(response)
    return 1.0 if choice == str(answer).upper() else 0.0


async def has_reasoning(completion) -> float:
    """Reward for providing reasoning."""
    if not completion:
        return 0.0
    response = completion[-1]["content"]
    return 0.2 if len(response) > 60 else 0.0


async def mentions_key_factors(completion, info) -> float:
    """Reward for considering important race factors."""
    if not completion:
        return 0.0
    response = completion[-1]["content"].lower()
    info_dict = _parse_info(info)

    score = 0.0
    if "tire" in response or "tyre" in response or "degrad" in response:
        score += 0.1
    if info_dict.get("rainfall", 0.0) > 0 and "rain" in response:
        score += 0.1
    if "gap" in response or "position" in response or "track position" in response:
        score += 0.1
    if info_dict.get("undercut_window") is not None and "undercut" in response:
        score += 0.05
    if info_dict.get("pit_loss_est") is not None and "pit" in response:
        score += 0.05
    if info_dict.get("sc_risk_proxy") is not None and ("safety car" in response or "vsc" in response):
        score += 0.05
    if "traffic" in response or "drs" in response:
        score += 0.05
    if "warmup" in response or "warm-up" in response:
        score += 0.05

    return min(score, 0.35)


async def acknowledges_uncertainty(completion) -> float:
    """Reward for recognizing uncertainty or tradeoffs."""
    if not completion:
        return 0.0
    response = completion[-1]["content"].lower()
    keywords = ["risk", "tradeoff", "uncertain", "if", "contingency", "depends"]
    return 0.1 if any(k in response for k in keywords) else 0.0


async def outcome_aligned(completion, answer, info) -> float:
    """Reward for alignment with actual race outcome."""
    if not completion:
        return 0.0
    info_dict = _parse_info(info)
    outcome_score = info_dict.get("outcome_score", 0.0)
    response = completion[-1]["content"]
    choice = _extract_final_choice(response)
    if choice == str(answer).upper():
        # Bonus if the outcome was positive (position gained or gap closed)
        return 0.15 if outcome_score > 0 else 0.05
    return 0.0


async def uses_tools(completion, tool_call_count: Optional[int] = None, use_tools: bool = False) -> float:
    """Reward for invoking at least one tool call. Only penalize when tools are enabled."""
    if tool_call_count and tool_call_count > 0:
        return 0.1
    # Only penalize missing tool calls when tools are explicitly enabled
    return -0.2 if use_tools else 0.0


async def final_choice_present(completion) -> float:
    """Penalty if a final choice is missing."""
    if not completion:
        return -0.3
    response = completion[-1]["content"]
    return 0.0 if _extract_final_choice(response) else -0.3


async def option_times_present(completion) -> float:
    """Reward for listing option time estimates in deep reasoning mode."""
    if not completion:
        return 0.0
    response = completion[-1]["content"]
    times = _extract_option_times(response)
    if len(times) >= 4:
        return 0.3
    if len(times) >= 2:
        return 0.1
    return 0.0


async def option_time_accuracy(completion, info) -> float:
    """Reward for accurate time estimates vs strategy model ground truth."""
    if not completion:
        return 0.0
    response = completion[-1]["content"]
    predicted = _extract_option_times(response)
    if not predicted:
        return 0.0
    info_dict = _parse_info(info)
    expected, _ = _expected_option_times(info_dict)
    shared = [k for k in predicted.keys() if k in expected]
    if not shared:
        return 0.0
    errors = [abs(predicted[k] - expected[k]) for k in shared]
    mae = sum(errors) / len(errors)
    # Scale reward: 0.6 max, linearly down to 0 at 8s MAE
    return max(0.0, 0.6 * (1.0 - (mae / 8.0)))


def load_environment(
    num_examples: int = -1,
    eval_season: Optional[int] = 2024,
    eval_tracks: Optional[List[str]] = None,
    use_tools: bool = False,
    multi_turn: bool = False,
    deep_reasoning: bool = True,
    multi_env: bool = False,
    env_tracks: Optional[List[str]] = None,
    max_tracks: int = 4,
    max_tokens: int = 900,
    env_id: str = "herr-professor/f1-strategy",
) -> vf.Environment:
    """Load the F1 strategy environment."""
    dataset = create_f1_dataset(use_tools=use_tools, multi_turn=multi_turn, deep_reasoning=deep_reasoning)
    train_dataset, eval_dataset = _split_train_eval(dataset, eval_season, eval_tracks)
    if train_dataset is None or len(train_dataset) == 0:
        train_dataset = dataset
    if eval_dataset is not None and len(eval_dataset) == 0:
        eval_dataset = None

    if num_examples > 0:
        train_dataset = train_dataset.select(range(min(num_examples, len(train_dataset))))
        if eval_dataset is not None:
            eval_dataset = eval_dataset.select(range(min(num_examples, len(eval_dataset))))

    def build_rubric(tools_enabled: bool = False) -> vf.Rubric:
        # Create closure to pass use_tools flag to uses_tools function
        async def uses_tools_wrapper(completion, tool_call_count: Optional[int] = None) -> float:
            return await uses_tools(completion, tool_call_count, use_tools=tools_enabled)

        if deep_reasoning:
            return vf.Rubric(
                funcs=[
                    correct_strategy,
                    final_choice_present,
                    option_times_present,
                    option_time_accuracy,
                    has_reasoning,
                    mentions_key_factors,
                    acknowledges_uncertainty,
                    outcome_aligned,
                    uses_tools_wrapper,
                ],
                weights=[1.0, 1.0, 0.3, 0.6, 0.2, 0.25, 0.1, 0.15, 0.1],
            )

        return vf.Rubric(
            funcs=[
                correct_strategy,
                has_reasoning,
                mentions_key_factors,
                acknowledges_uncertainty,
                outcome_aligned,
                uses_tools_wrapper,
                final_choice_present,
            ],
            weights=[1.0, 0.2, 0.35, 0.1, 0.15, 0.1, 1.0],
        )

    def build_env(ds: Dataset, eval_ds: Optional[Dataset]) -> vf.Environment:
        rubric = build_rubric(tools_enabled=use_tools)
        if use_tools:
            return F1ToolEnv(dataset=ds, eval_dataset=eval_ds, rubric=rubric, max_tokens=max_tokens, max_turns=6)
        if multi_turn:
            return F1PitWallEnv(dataset=ds, eval_dataset=eval_ds, rubric=rubric, max_tokens=max_tokens, max_turns=2)
        return vf.SingleTurnEnv(dataset=ds, eval_dataset=eval_ds, rubric=rubric, max_tokens=max_tokens)

    if not multi_env:
        return build_env(train_dataset, eval_dataset)

    tracks = env_tracks
    if not tracks:
        all_tracks = train_dataset["track"] if "track" in train_dataset.column_names else []
        counts: Dict[str, int] = {}
        for t in all_tracks:
            if t is None:
                continue
            counts[t] = counts.get(t, 0) + 1
        tracks = [t for t, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)][:max_tracks]

    envs = []
    env_names = []

    for track in tracks:
        track_ds = train_dataset.filter(lambda row: row.get("track") == track)
        track_eval = None
        if eval_dataset is not None:
            filtered_eval = eval_dataset.filter(lambda row: row.get("track") == track)
            # Only use if non-empty; EnvGroup requires formatted datasets
            track_eval = filtered_eval if len(filtered_eval) > 0 else None
        if len(track_ds) == 0:
            continue
        envs.append(build_env(track_ds, track_eval))
        env_names.append(track)

    return F1EnvGroup(envs=envs, env_names=env_names, env_id=env_id)
