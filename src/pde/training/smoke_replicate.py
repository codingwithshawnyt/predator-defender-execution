"""Phase-1 smoke test: replicate Lillo & Macri (2024) symmetric two-liquidator result.

Reference: arXiv:2408.11773

Trains two independent DQN agents (one per liquidator) in a shared
PyMarketSim LOB environment.  Each agent observes (time_left,
remaining_inventory, fundamental, best_bid, best_ask) and selects a
discrete volume bin.  The training loop alternates between collecting
experience from both agents and updating their policies.

Usage
-----
    uv run python -m pde.training.smoke_replicate [--n-steps 100] [--episodes 5000] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN

from pde.envs.symmetric_env import SymmetricLiquidatorEnv


def compute_is(liq, arrival_price: float, final_fund: float) -> float:
    """Implementation shortfall for a liquidator.

    IS = (arrival_price * q_0) - (cash_received + remaining * final_fund)
    Positive IS means the liquidator paid more than the arrival benchmark.
    """
    cash = liq.cash
    remaining = liq.remaining_inventory
    total_value = cash + remaining * final_fund
    benchmark = arrival_price * liq.q_0
    return benchmark - total_value


def train(
    n_steps: int = 100,
    q_0: float = 100.0,
    n_bg: int = 50,
    alpha: float = 0.002,
    n_volume_bins: int = 21,
    total_episodes: int = 5000,
    learning_rate: float = 1e-4,
    batch_size: int = 64,
    buffer_size: int = 15_000,
    gamma: float = 1.0,
    learning_starts: int = 500,
    exploration_fraction: float = 0.6,
    exploration_final_eps: float = 0.05,
    target_update_interval: int = 500,
    seed: int = 0,
    log_dir: str | None = None,
    eval_interval: int = 100,
    n_eval_episodes: int = 20,
    device: str = "auto",
) -> dict:
    """Train two DQN liquidators in the symmetric environment.

    Returns a dict of training metrics for downstream plotting.
    """
    if log_dir is None:
        log_dir = str(Path("artifacts/logs/smoke_replicate"))
    os.makedirs(log_dir, exist_ok=True)

    output_dir = Path(log_dir)

    # ---- Build env ----
    env = SymmetricLiquidatorEnv(
        n_steps=n_steps,
        q_0=q_0,
        n_background_agents=n_bg,
        alpha=alpha,
        n_volume_bins=n_volume_bins,
        normalizers={"fundamental": 100_000.0},
        seed=seed,
    )

    # ---- Build one DQN per agent ----
    # We create a minimal single-agent Gym env so SB3 DQN can

    agent_nets: dict[str, DQN] = {}
    for aid in env.possible_agents:

        class _SingleAgentEnv(gym.Env):  # type: ignore[misc]
            observation_space = env.observation_spaces[aid]
            action_space = env.action_spaces[aid]

            def reset(self, *, seed=None, options=None):
                return self.observation_space.sample(), {}

            def step(self, action):
                obs = self.observation_space.sample()
                return obs, 0.0, True, True, {}

        agent_nets[aid] = DQN(
            policy="MlpPolicy",
            env=_SingleAgentEnv(),
            learning_rate=learning_rate,
            batch_size=batch_size,
            buffer_size=buffer_size,
            gamma=gamma,
            learning_starts=learning_starts,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            target_update_interval=target_update_interval,
            policy_kwargs={"net_arch": [30, 30, 30, 30, 30]},
            seed=seed + hash(aid) % 10000,
            verbose=0,
            device=device,
        )

    # ---- Training loop ----
    episode_rewards: dict[str, list[float]] = {"liq_0": [], "liq_1": []}
    episode_is: dict[str, list[float]] = {"liq_0": [], "liq_1": []}
    eval_rewards: list[dict] = []
    global_step = 0

    for ep in range(total_episodes):
        obs_dict, _ = env.reset(seed=seed + ep)
        ep_reward = {"liq_0": 0.0, "liq_1": 0.0}
        arrival_fund = env.market.fundamental.get_value_at(0)
        done = False

    while not done:
        # Select actions (epsilon-greedy per agent)
        actions = {}
        active_agents = list(env.agents)
        for aid in active_agents:
            dqn = agent_nets[aid]
            obs_tensor = (
                torch.as_tensor(obs_dict[aid], dtype=torch.float32).unsqueeze(0).to(dqn.device)
            )
            if global_step < dqn.learning_starts:
                actions[aid] = np.random.randint(0, n_volume_bins)
            else:
                with torch.no_grad():
                    q_vals = dqn.q_network(obs_tensor)  # type: ignore[attr-defined]
                if np.random.random() < dqn.exploration_rate:
                    actions[aid] = np.random.randint(0, n_volume_bins)
                else:
                    actions[aid] = int(q_vals.argmax(dim=1).item())

        # Step environment
        next_obs, rewards, terminated, truncated, _infos = env.step(actions)

        # Store transitions in each agent's replay buffer
        for aid in active_agents:
            dqn = agent_nets[aid]
            action_arr = np.array([actions[aid]], dtype=np.int64)
            reward_arr = np.array([rewards[aid]], dtype=np.float32)
            done_arr = np.array([terminated[aid] or truncated[aid]], dtype=np.float32)
            obs_arr = obs_dict[aid].reshape(1, -1)
            next_obs_arr = next_obs[aid].reshape(1, -1)

            dqn.replay_buffer.add(  # type: ignore[union-attr]
                obs=obs_arr,
                action=action_arr,
                reward=reward_arr,
                next_obs=next_obs_arr,
                done=done_arr,
                infos=[{}],
            )
        for aid in active_agents:
            ep_reward[aid] += rewards[aid]

        obs_dict = next_obs
        global_step += 1

        # Gradient updates
        for _aid, dqn in agent_nets.items():
            if global_step > dqn.learning_starts and global_step % 4 == 0:
                dqn.train(batch_size=batch_size)  # type: ignore[call-arg]

        if all(terminated.values()) or all(truncated.values()):
            done = True

        # Episode IS
        final_fund = env.market.get_final_fundamental()
        for aid in env.possible_agents:
            liq = env._liquidators[aid]
            is_val = compute_is(liq, arrival_fund, final_fund)
            episode_rewards[aid].append(ep_reward[aid])
            episode_is[aid].append(is_val)

        # Periodic evaluation
        if (ep + 1) % eval_interval == 0:
            mean_r0 = np.mean(episode_rewards["liq_0"][-eval_interval:])
            mean_r1 = np.mean(episode_rewards["liq_1"][-eval_interval:])
            mean_is0 = np.mean(episode_is["liq_0"][-eval_interval:])
            mean_is1 = np.mean(episode_is["liq_1"][-eval_interval:])
            eval_rewards.append(
                {
                    "episode": ep + 1,
                    "mean_reward_liq_0": mean_r0,
                    "mean_reward_liq_1": mean_r1,
                    "mean_is_liq_0": mean_is0,
                    "mean_is_liq_1": mean_is1,
                }
            )
            print(
                f"Ep {ep+1:5d} | "
                f"R0={mean_r0:+.2f} R1={mean_r1:+.2f} | "
                f"IS0={mean_is0:+.2f} IS1={mean_is1:+.2f}"
            )

    # ---- Save results ----
    results = {
        "config": {
            "n_steps": n_steps,
            "q_0": q_0,
            "n_bg": n_bg,
            "alpha": alpha,
            "n_volume_bins": n_volume_bins,
            "total_episodes": total_episodes,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "gamma": gamma,
            "seed": seed,
        },
        "episode_rewards_liq_0": episode_rewards["liq_0"],
        "episode_rewards_liq_1": episode_rewards["liq_1"],
        "episode_is_liq_0": episode_is["liq_0"],
        "episode_is_liq_1": episode_is["liq_1"],
        "eval_log": eval_rewards,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    # Save models
    models_dir = Path("artifacts/checkpoints/smoke_replicate")
    models_dir.mkdir(parents=True, exist_ok=True)
    for aid, dqn in agent_nets.items():
        dqn.save(str(models_dir / f"dqn_{aid}"))
    print(f"Models saved to {models_dir}/")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Lillo-Macri smoke replicate")
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--q-0", type=float, default=100.0)
    parser.add_argument("--n-bg", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.002)
    parser.add_argument("--n-volume-bins", type=int, default=21)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=15_000)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train(
        n_steps=args.n_steps,
        q_0=args.q_0,
        n_bg=args.n_bg,
        alpha=args.alpha,
        n_volume_bins=args.n_volume_bins,
        total_episodes=args.episodes,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        gamma=args.gamma,
        seed=args.seed,
        log_dir=args.log_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
