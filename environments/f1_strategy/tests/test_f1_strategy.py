#!/usr/bin/env python3
"""Tests for f1_strategy environment."""
import asyncio
import pytest


def test_environment_loads():
    """Test that the environment loads without errors."""
    from f1_strategy import load_environment
    env = load_environment(eval_season=None, deep_reasoning=True)  # No eval split, only 2023 data
    assert env is not None
    assert len(env.dataset) > 0


def test_multi_turn_env_loads():
    """Test multi-turn pit wall environment."""
    from f1_strategy import load_environment
    env = load_environment(multi_turn=True, eval_season=None, deep_reasoning=True)
    assert type(env).__name__ == "F1PitWallEnv"


def test_tool_env_loads():
    """Test tool environment with all three tools."""
    from f1_strategy import load_environment
    env = load_environment(use_tools=True, eval_season=None, deep_reasoning=True)
    assert type(env).__name__ == "F1ToolEnv"
    assert hasattr(env, "tire_deg_estimator")
    assert hasattr(env, "pit_delta_lookup")
    assert hasattr(env, "weather_confidence")


def test_multi_env_loads():
    """Test multi-environment mode creates EnvGroup."""
    from f1_strategy import load_environment
    env = load_environment(multi_env=True, max_tracks=2, eval_season=None, deep_reasoning=True)
    assert type(env).__name__ in ("EnvGroup", "F1EnvGroup")
    assert len(env.envs) <= 2


def test_correct_strategy_rubric():
    """Test correct_strategy reward function."""
    from f1_strategy import correct_strategy
    
    async def _test():
        completion = [{"role": "assistant", "content": "C) Stay out on current tires."}]
        assert await correct_strategy(completion, "C") == 1.0
        assert await correct_strategy(completion, "A") == 0.0
        assert await correct_strategy([], "C") == 0.0
    
    asyncio.run(_test())


def test_has_reasoning_rubric():
    """Test has_reasoning reward function."""
    from f1_strategy import has_reasoning
    
    async def _test():
        long = [{"role": "assistant", "content": "C) Stay out because " + "x" * 100}]
        short = [{"role": "assistant", "content": "C)"}]
        assert await has_reasoning(long) == 0.2
        assert await has_reasoning(short) == 0.0
    
    asyncio.run(_test())


def test_track_priors():
    """Test track priors exist for key tracks."""
    from f1_strategy import TRACK_PRIORS, _track_profile
    
    # Check known tracks
    for track in ["Monaco", "Jeddah", "Melbourne", "Sakhir"]:
        profile = _track_profile(track)
        assert "pit_loss" in profile
        assert "overtake_difficulty" in profile
        assert "sc_risk_base" in profile
    
    # Unknown track should get default
    default = _track_profile("UnknownTrack")
    assert default["pit_loss"] == 22.0


def test_deep_reasoning_prompt_includes_model_block():
    """Ensure deep reasoning prompts include the strategy model block."""
    from f1_strategy import _build_prompt_text
    info = {
        "track": "Monaco",
        "tire_compound": "medium",
        "lap_duration": 95.0,
        "degradation_slope": 0.05,
        "undercut_window": 20.0,
        "pit_loss_est": 22.0,
        "traffic_tightness": 1.0,
        "sc_risk_proxy": 0.3,
        "rainfall": 0.0,
        "rain_soon": False,
    }
    prompt = _build_prompt_text(info, use_tools=False, multi_turn=False, deep_reasoning=True)
    assert "Strategy Model" in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
