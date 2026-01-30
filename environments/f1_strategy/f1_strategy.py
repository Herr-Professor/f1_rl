import json
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


def _track_profile(track: str) -> Dict[str, float]:
    return TRACK_PRIORS.get(track, DEFAULT_TRACK_PROFILE)


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
        "You may use tools for tire degradation, pit delta, or weather confidence."
        if use_tools
        else "Use the provided signals; do not invent data."
    )
    follow_up = (
        "The pit wall may ask one follow-up question. Answer it concisely and restate your decision."
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


def _build_prompt_text(info: Dict[str, Any], use_tools: bool, multi_turn: bool) -> str:
    track = info.get("track", "Unknown")
    profile = _track_profile(track)
    weather_summary = "Rain at circuit" if info.get("rainfall", 0.0) > 0 else "Clear, no changes expected"

    signals = [
        f"Pace Delta: {_format_signal(info.get('pace_delta'), 's')} (last 3 laps vs stint median)",
        f"Degradation Slope: {_format_signal(info.get('degradation_slope'), ' s/lap')}",
        f"Undercut Window: {_format_signal(info.get('undercut_window'), 's')}",
        f"Estimated Pit Loss: {_format_signal(info.get('pit_loss_est') or profile['pit_loss'], 's')}",
        f"Safety Car Risk: {_format_signal(info.get('sc_risk_proxy'), '')} (base {profile['sc_risk_base']})",
    ]

    context = f"""Race Status - Lap {info.get('lap', '?')}/{info.get('total_laps', '?')}
Position: P{info.get('position', '?')}
Current Tires: {str(info.get('tire_compound', 'medium')).capitalize()} ({info.get('tire_age', '?')} laps old)
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

    decision = """

What is your strategy?
A) Pit now for soft tires
B) Pit now for intermediates (prepare for rain)
C) Stay out on current tires
D) Pit when rain starts

Reply with your choice and reasoning."""

    if multi_turn:
        decision += " If you need clarification, answer the pit wall question when asked."

    return f"{context}\n\n{priors}{decision}"


def _build_prompt_messages(info: Dict[str, Any], use_tools: bool, multi_turn: bool) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": _system_prompt(use_tools, multi_turn)},
        {"role": "user", "content": _build_prompt_text(info, use_tools, multi_turn)},
    ]


def _prepare_dataset(dataset: Dataset, use_tools: bool, multi_turn: bool) -> Dataset:
    def _map(row, idx):
        info = _parse_info(row.get("info"))
        info.setdefault("track", row.get("track"))
        info.setdefault("season", row.get("season"))
        row["info"] = json.dumps(info)
        row["track"] = info.get("track")
        row["season"] = info.get("season")
        row["prompt"] = _build_prompt_messages(info, use_tools, multi_turn)
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


def create_f1_dataset(use_tools: bool = False, multi_turn: bool = False) -> Dataset:
    """Create F1 racing strategy scenarios from OpenF1 cache."""
    openf1_scenarios = _load_openf1_scenarios()
    if not openf1_scenarios:
        raise RuntimeError(
            "OpenF1 scenarios not found. Run scripts/build_openf1_dataset.py to generate data/openf1_scenarios.jsonl."
        )
    dataset = Dataset.from_list(openf1_scenarios)
    return _prepare_dataset(dataset, use_tools, multi_turn)


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

    def update_tool_args(self, tool_name, tool_args, state):
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


async def correct_strategy(completion, answer) -> float:
    """Reward for choosing the correct strategy."""
    import re
    if not completion:
        return 0.0
    response = completion[-1]["content"].upper()
    # Match patterns like "A)", "A.", "A:", "A ", standalone "A" at start, or "OPTION A"
    pattern = rf"(?:^|\s|OPTION\s*){answer}(?:[).:,\s]|$)"
    return 1.0 if re.search(pattern, response) else 0.0


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

    return min(score, 0.3)


async def acknowledges_uncertainty(completion) -> float:
    """Reward for recognizing uncertainty or tradeoffs."""
    if not completion:
        return 0.0
    response = completion[-1]["content"].lower()
    keywords = ["risk", "tradeoff", "uncertain", "if", "contingency", "depends"]
    return 0.1 if any(k in response for k in keywords) else 0.0


def load_environment(
    num_examples: int = -1,
    eval_season: Optional[int] = 2024,
    eval_tracks: Optional[List[str]] = None,
    use_tools: bool = False,
    multi_turn: bool = False,
    multi_env: bool = False,
    env_tracks: Optional[List[str]] = None,
    max_tracks: int = 4,
    max_tokens: int = 512,
) -> vf.Environment:
    """Load the F1 strategy environment."""
    dataset = create_f1_dataset(use_tools=use_tools, multi_turn=multi_turn)
    train_dataset, eval_dataset = _split_train_eval(dataset, eval_season, eval_tracks)

    if num_examples > 0:
        train_dataset = train_dataset.select(range(min(num_examples, len(train_dataset))))
        if eval_dataset is not None:
            eval_dataset = eval_dataset.select(range(min(num_examples, len(eval_dataset))))

    def build_rubric() -> vf.Rubric:
        return vf.Rubric(
            funcs=[correct_strategy, has_reasoning, mentions_key_factors, acknowledges_uncertainty],
            weights=[1.0, 0.2, 0.3, 0.1],
        )

    def build_env(ds: Dataset, eval_ds: Optional[Dataset]) -> vf.Environment:
        rubric = build_rubric()
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

    return vf.EnvGroup(envs=envs, env_names=env_names)
