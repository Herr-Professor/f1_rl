#!/usr/bin/env python3
"""Build OpenF1-based scenarios for the f1_strategy environment."""
import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://api.openf1.org/v1"
USER_AGENT = "f1-strategy-dataset-builder/0.1"


@dataclass
class SessionInfo:
    session_key: int
    meeting_key: int
    session_name: str
    date_start: str
    circuit: str


def _parse_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    # OpenF1 uses ISO8601 with Z
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _request_json(path: str, params: Optional[List[Tuple[str, Any]]] = None) -> List[Dict[str, Any]]:
    query = ""
    if params:
        query = "?" + urlencode(params, doseq=True)
    url = f"{BASE_URL}{path}{query}"

    last_err: Optional[Exception] = None
    for attempt in range(5):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=60) as resp:
                data = resp.read().decode("utf-8")
            return json.loads(data)
        except (HTTPError, URLError, TimeoutError) as err:
            last_err = err
            time.sleep(0.5 * (2**attempt))
            continue

    if last_err:
        raise last_err
    return []


def _sleep_between_requests():
    time.sleep(0.2)


def _fetch_sessions(year: int) -> List[SessionInfo]:
    sessions = _request_json(
        "/sessions",
        [
            ("session_name", "Race"),
        ],
    )
    _sleep_between_requests()

    results: List[SessionInfo] = []
    for s in sessions:
        if not s.get("session_key") or not s.get("meeting_key"):
            continue
        date_start = s.get("date_start", "")
        dt = _parse_datetime(date_start)
        if not dt or dt.year != year:
            continue
        circuit = s.get("circuit_short_name") or s.get("location") or s.get("country_name") or "Unknown"
        results.append(
            SessionInfo(
                session_key=int(s["session_key"]),
                meeting_key=int(s["meeting_key"]),
                session_name=s.get("session_name", "Race"),
                date_start=date_start,
                circuit=circuit,
            )
        )
    return results


def _fetch_stints(session_key: int) -> List[Dict[str, Any]]:
    stints = _request_json("/stints", [("session_key", session_key)])
    _sleep_between_requests()
    return stints


def _fetch_laps(session_key: int, driver_number: int) -> List[Dict[str, Any]]:
    laps = _request_json(
        "/laps",
        [
            ("session_key", session_key),
            ("driver_number", driver_number),
        ],
    )
    _sleep_between_requests()
    return laps


def _fetch_positions(session_key: int, driver_number: int) -> List[Dict[str, Any]]:
    positions = _request_json(
        "/position",
        [
            ("session_key", session_key),
            ("driver_number", driver_number),
        ],
    )
    _sleep_between_requests()
    return positions


def _fetch_intervals(session_key: int, driver_number: int) -> List[Dict[str, Any]]:
    intervals = _request_json(
        "/intervals",
        [
            ("session_key", session_key),
            ("driver_number", driver_number),
        ],
    )
    _sleep_between_requests()
    return intervals


def _fetch_weather(session_key: int) -> List[Dict[str, Any]]:
    weather = _request_json("/weather", [("session_key", session_key)])
    _sleep_between_requests()
    return weather


def _fetch_pits(session_key: int) -> List[Dict[str, Any]]:
    pits = _request_json("/pit", [("session_key", session_key)])
    _sleep_between_requests()
    return pits


def _fetch_race_control(session_key: int) -> List[Dict[str, Any]]:
    control = _request_json("/race_control", [("session_key", session_key)])
    _sleep_between_requests()
    return control


def _nearest_by_date(rows: Iterable[Dict[str, Any]], target: datetime) -> Optional[Dict[str, Any]]:
    best = None
    best_delta = None
    for row in rows:
        dt = _parse_datetime(row.get("date") or row.get("date_start"))
        if not dt:
            continue
        delta = abs((dt - target).total_seconds())
        if best_delta is None or delta < best_delta:
            best = row
            best_delta = delta
    return best


def _linear_slope(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2 == 0:
        return (values[mid - 1] + values[mid]) / 2.0
    return values[mid]


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_prompt(
    lap_number: int,
    total_laps: int,
    position: int,
    compound: str,
    tire_age: int,
    gap_to_leader: float,
    interval: float,
    fuel_remaining: int,
    weather_line: str,
    pace_delta: float,
    degradation_slope: float,
    undercut_window: float,
    pit_loss_est: float,
    sc_risk_proxy: float,
    track: str,
) -> str:
    return f"""Race Status - Lap {lap_number}/{total_laps}
Position: P{position}
Current Tires: {compound.capitalize()} ({tire_age} laps old)
Gap to Leader: +{gap_to_leader:.1f}s
Gap to Car Ahead: +{interval:.1f}s
Fuel: {fuel_remaining}% remaining
Weather: {weather_line}
Pace Delta (last 3 laps vs stint median): {pace_delta:+.2f}s
Degradation Slope (s/lap): {degradation_slope:+.3f}
Undercut Window (s): {undercut_window:+.1f}
Estimated Pit Loss (s): {pit_loss_est:.1f}
Safety Car Risk (0-1): {sc_risk_proxy:.2f}
Track: {track}

What is your strategy?
A) Pit now for soft tires
B) Pit now for intermediates (prepare for rain)
C) Stay out on current tires
D) Pit when rain starts

Reply with your choice and reasoning.
"""


def _label_strategy(
    laps_to_pit: int,
    compound: str,
    tire_age: int,
    rainfall: float,
) -> str:
    if rainfall > 0:
        if laps_to_pit <= 2:
            return "B"
        return "D"

    if laps_to_pit <= 2:
        return "A"
    if compound == "hard" and tire_age > 20:
        return "A"
    if tire_age < 10:
        return "C"
    return "C"


def build_dataset(
    years: List[int],
    max_sessions: int,
    max_scenarios: int,
    out_path: Path,
) -> int:
    sessions: List[SessionInfo] = []
    for year in years:
        sessions.extend(_fetch_sessions(year))
    sessions = sorted(sessions, key=lambda s: s.date_start)[:max_sessions]

    scenarios: List[Dict[str, Any]] = []
    for session in sessions:
        stints = _fetch_stints(session.session_key)
        if not stints:
            continue

        total_laps = max((s.get("lap_end") or 0) for s in stints) or 60
        weather_rows = _fetch_weather(session.session_key)
        pits = _fetch_pits(session.session_key)
        race_control = _fetch_race_control(session.session_key)
        pit_durations = [
            _safe_float(p.get("pit_duration")) for p in pits if _safe_float(p.get("pit_duration"))
        ]
        pit_loss_est = _median(pit_durations) if pit_durations else 22.0

        positions_cache: Dict[int, List[Dict[str, Any]]] = {}
        intervals_cache: Dict[int, List[Dict[str, Any]]] = {}
        laps_cache: Dict[int, List[Dict[str, Any]]] = {}

        for stint in stints:
            if len(scenarios) >= max_scenarios:
                break

            driver_number = stint.get("driver_number")
            lap_start = stint.get("lap_start")
            lap_end = stint.get("lap_end")
            compound = stint.get("compound") or "medium"
            tyre_age_at_start = stint.get("tyre_age_at_start") or 0

            if not driver_number or not lap_start or not lap_end:
                continue

            lap_number = int(lap_start + (lap_end - lap_start) // 2)
            if driver_number not in laps_cache:
                laps_cache[int(driver_number)] = _fetch_laps(session.session_key, int(driver_number))
            laps = laps_cache[int(driver_number)]
            lap = next((l for l in laps if int(l.get("lap_number", -1)) == lap_number), None)
            if not lap:
                continue

            lap_date = _parse_datetime(lap.get("date_start") or lap.get("date"))
            if not lap_date:
                continue

            if driver_number not in positions_cache:
                positions_cache[int(driver_number)] = _fetch_positions(session.session_key, int(driver_number))
            if driver_number not in intervals_cache:
                intervals_cache[int(driver_number)] = _fetch_intervals(session.session_key, int(driver_number))

            position_row = _nearest_by_date(positions_cache[int(driver_number)], lap_date)
            interval_row = _nearest_by_date(intervals_cache[int(driver_number)], lap_date)
            weather_row = _nearest_by_date(weather_rows, lap_date) if weather_rows else None

            if not position_row or not interval_row:
                continue

            position = position_row.get("position")
            gap_to_leader = interval_row.get("gap_to_leader")
            interval = interval_row.get("interval")

            if position is None or gap_to_leader is None or interval is None:
                continue

            rainfall = float(weather_row.get("rainfall", 0.0)) if weather_row else 0.0
            track_temp = _safe_float(weather_row.get("track_temp")) if weather_row else None
            air_temp = _safe_float(weather_row.get("air_temp")) if weather_row else None
            humidity = _safe_float(weather_row.get("humidity")) if weather_row else None
            wind_speed = _safe_float(weather_row.get("wind_speed")) if weather_row else None
            weather_line = "Rain at circuit" if rainfall > 0 else "Clear, no changes expected"

            stint_laps = [
                l for l in laps
                if l.get("lap_number") is not None
                and int(l["lap_number"]) >= int(lap_start)
                and int(l["lap_number"]) <= int(lap_end)
            ]
            lap_durations = [
                _safe_float(l.get("lap_duration")) for l in stint_laps if _safe_float(l.get("lap_duration"))
            ]
            lap_duration = _safe_float(lap.get("lap_duration")) or 0.0
            last3 = lap_durations[-3:] if len(lap_durations) >= 3 else lap_durations
            pace_delta = lap_duration - (_median(last3) if last3 else lap_duration)

            xs = list(range(len(lap_durations)))
            degradation_slope = _linear_slope(xs, lap_durations)

            first3 = lap_durations[:3] if len(lap_durations) >= 3 else lap_durations
            fresh_gain = max(0.0, (_median(last3) - _median(first3)) if first3 and last3 else 0.0)
            undercut_window = pit_loss_est - fresh_gain

            sc_events = [
                r for r in race_control
                if _parse_datetime(r.get("date"))
                and abs((_parse_datetime(r.get("date")) - lap_date).total_seconds()) < 900
                and (
                    "SAFETY" in (r.get("message", "") or "").upper()
                    or "VSC" in (r.get("message", "") or "").upper()
                    or "SAFETY" in (r.get("category", "") or "").upper()
                )
            ]
            sc_risk_proxy = min(1.0, len(sc_events) / 3.0)

            tire_age = int(tyre_age_at_start + max(0, lap_number - lap_start))
            laps_to_pit = int(lap_end - lap_number)
            fuel_remaining = max(0, min(100, int(round((1 - (lap_number / max(total_laps, 1))) * 100))))

            answer = _label_strategy(laps_to_pit, compound.lower(), tire_age, rainfall)

            prompt_text = _build_prompt(
                lap_number=lap_number,
                total_laps=total_laps,
                position=int(position),
                compound=compound.lower(),
                tire_age=tire_age,
                gap_to_leader=float(gap_to_leader),
                interval=float(interval),
                fuel_remaining=fuel_remaining,
                weather_line=weather_line,
                pace_delta=pace_delta,
                degradation_slope=degradation_slope,
                undercut_window=undercut_window,
                pit_loss_est=pit_loss_est,
                sc_risk_proxy=sc_risk_proxy,
                track=session.circuit,
            )

            scenarios.append(
                {
                    "prompt": [
                        {
                            "role": "system",
                            "content": (
                                "You are an F1 race strategist. Analyze the race "
                                "situation and recommend the best strategy."
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt_text,
                        },
                    ],
                    "answer": answer,
                    "info": json.dumps(
                        {
                            "source": "openf1",
                            "session_key": session.session_key,
                            "meeting_key": session.meeting_key,
                            "season": _parse_datetime(session.date_start).year if _parse_datetime(session.date_start) else None,
                            "driver_number": int(driver_number),
                            "lap": lap_number,
                            "total_laps": total_laps,
                            "track": session.circuit,
                            "position": int(position),
                            "tire_compound": compound.lower(),
                            "tire_age": tire_age,
                            "fuel_remaining": fuel_remaining,
                            "gap_to_leader": float(gap_to_leader),
                            "interval": float(interval),
                            "lap_duration": float(lap_duration),
                            "pace_delta": float(pace_delta),
                            "degradation_slope": float(degradation_slope),
                            "undercut_window": float(undercut_window),
                            "pit_loss_est": float(pit_loss_est),
                            "sc_risk_proxy": float(sc_risk_proxy),
                            "rainfall": rainfall,
                            "track_temp": track_temp,
                            "air_temp": air_temp,
                            "humidity": humidity,
                            "wind_speed": wind_speed,
                        }
                    ),
                }
            )

        if len(scenarios) >= max_scenarios:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in scenarios:
            f.write(json.dumps(item) + "\n")

    return len(scenarios)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OpenF1 dataset for f1_strategy.")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    parser.add_argument("--max-sessions", type=int, default=6)
    parser.add_argument("--max-scenarios", type=int, default=200)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "openf1_scenarios.jsonl",
    )
    args = parser.parse_args()

    count = build_dataset(
        years=args.years,
        max_sessions=args.max_sessions,
        max_scenarios=args.max_scenarios,
        out_path=args.out,
    )
    print(f"Wrote {count} scenarios to {args.out}")


if __name__ == "__main__":
    main()
