"""Tests for LiquidatorAgent and SymmetricLiquidatorEnv."""

from __future__ import annotations

import pytest


def test_liquidator_agent_import():
    from pde.agents.liquidator import LiquidatorAgent

    assert LiquidatorAgent is not None


def test_symmetric_env_import():
    from pde.envs.symmetric_env import SymmetricLiquidatorEnv

    assert SymmetricLiquidatorEnv is not None


def test_symmetric_env_spaces():
    from pde.envs.symmetric_env import SymmetricLiquidatorEnv

    env = SymmetricLiquidatorEnv(n_steps=10, q_0=10.0, n_background_agents=5)
    assert "liq_0" in env.observation_spaces
    assert "liq_1" in env.observation_spaces
    assert env.observation_spaces["liq_0"].shape == (5,)
    assert env.action_spaces["liq_0"].n == 21


def test_symmetric_env_reset():
    from pde.envs.symmetric_env import SymmetricLiquidatorEnv

    env = SymmetricLiquidatorEnv(n_steps=10, q_0=10.0, n_background_agents=5, seed=42)
    obs, _infos = env.reset(seed=42)
    assert "liq_0" in obs
    assert "liq_1" in obs
    assert obs["liq_0"].shape == (5,)
    assert obs["liq_1"].shape == (5,)
    assert obs["liq_0"][1] == 1.0


def test_symmetric_env_step():
    from pde.envs.symmetric_env import SymmetricLiquidatorEnv

    env = SymmetricLiquidatorEnv(n_steps=10, q_0=10.0, n_background_agents=5, seed=42)
    env.reset(seed=42)
    actions = {"liq_0": 10, "liq_1": 10}
    obs, rewards, _terminated, _truncated, _infos = env.step(actions)
    assert "liq_0" in obs
    assert "liq_1" in rewards
    assert isinstance(rewards["liq_0"], float)


def test_symmetric_env_full_episode():
    from pde.envs.symmetric_env import SymmetricLiquidatorEnv

    env = SymmetricLiquidatorEnv(n_steps=10, q_0=10.0, n_background_agents=5, seed=42)
    env.reset(seed=42)
    done = False
    steps = 0
    while not done:
        actions = dict.fromkeys(env.agents, 2)
        _obs, _rewards, terminated, truncated, _infos = env.step(actions)
        steps += 1
        if all(terminated.values()) or all(truncated.values()):
            done = True
        if steps > 20:
            break
    assert steps > 0


def test_action_to_volume_mapping():
    from pde.envs.symmetric_env import SymmetricLiquidatorEnv

    env = SymmetricLiquidatorEnv(n_steps=10, q_0=100.0, n_volume_bins=21, n_background_agents=5)
    env.reset(seed=0)
    vol_0 = env._action_to_volume("liq_0", 0)
    assert vol_0 == 0.0
    vol_20 = env._action_to_volume("liq_0", 20)
    assert vol_20 == pytest.approx(100.0)
    vol_10 = env._action_to_volume("liq_0", 10)
    assert vol_10 == pytest.approx(50.0)
