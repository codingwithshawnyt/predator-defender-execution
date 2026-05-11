"""SymmetricLiquidatorEnv: PettingZoo ParallelEnv for two RL liquidators.

Replicates Lillo & Macri (2024) symmetric two-liquidator experiment
on top of PyMarketSim's LOB simulator.

Time convention: 1 sim step = 1 second.  Both liquidators arrive
every step (lam = 1.0).  Background ZI traders arrive stochastically.

Observation (per agent, 5-dim):
    [time_left / N, remaining_inventory / q_0,
     fundamental / norm, best_bid / norm, best_ask / norm]

Action (per agent, discrete):
    Integer in {0, 1, ..., K-1} where K = n_volume_bins.
    Maps to volume = (action / (K-1)) * remaining_inventory.

Reward (per agent, per step):
    r_t = cash_from_execution - alpha * volume^2
    (Implementation-shortfall style, matching Lillo-Macri Eq. 23.)
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import ClassVar

import numpy as np
import torch
import torch.distributions as dist
from gymnasium import spaces
from marketsim.agent.noise_ZI_agent import ZIAgent
from marketsim.fourheap.constants import BUY, SELL
from marketsim.fundamental.lazy_mean_reverting import LazyGaussianMeanReverting
from marketsim.market.market import Market
from pettingzoo import ParallelEnv

from pde.agents.liquidator import LiquidatorAgent


def _sample_arrivals(p: float, n: int) -> torch.Tensor:
    return dist.Geometric(torch.tensor([p])).sample((n,)).squeeze()


class SymmetricLiquidatorEnv(ParallelEnv):
    """Two symmetric liquidators in a PyMarketSim LOB.

    Parameters
    ----------
    n_steps : int
        Episode horizon N (each agent acts N times).
    q_0 : float
        Initial inventory per liquidator.
    n_background_agents : int
        Number of ZI noise traders.
    lam_bg : float
        Arrival rate parameter for background traders.
    mean : float
        Fundamental mean level.
    r : float
        Mean-reversion rate for fundamental.
    shock_var : float
        Fundamental shock variance.
    q_max : int
        Max position for ZI agents (PrivateValues).
    pv_var : float
        Private-values variance for ZI agents.
    shade : list[float] | None
        ZI agent shade [low, high].
    alpha : float
        Temporary-impact quadratic penalty coefficient.
    n_volume_bins : int
        Number of discrete volume bins K (action = 0..K-1).
    normalizers : dict | None
        Observation normalizers.  Must contain ``fundamental``.
    seed : int | None
        RNG seed for reproducibility.
    """

    metadata: ClassVar = {"name": "symmetric_liquidator_v0", "render_modes": []}

    # PettingZoo agent IDs
    possible_agents: ClassVar = ["liq_0", "liq_1"]

    def __init__(
        self,
        n_steps: int = 100,
        q_0: float = 100.0,
        n_background_agents: int = 50,
        lam_bg: float = 0.1,
        mean: float = 100_000.0,
        r: float = 0.05,
        shock_var: float = 1e6,
        q_max: int = 10,
        pv_var: float = 5e6,
        shade: list[float] | None = None,
        alpha: float = 0.002,
        n_volume_bins: int = 21,
        normalizers: dict | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.n_steps = n_steps
        self.q_0 = q_0
        self.n_bg = n_background_agents
        self.lam_bg = lam_bg
        self.mean = mean
        self.r = r
        self.shock_var = shock_var
        self.q_max = q_max
        self.pv_var = pv_var
        self.shade = shade or [10, 30]
        self.alpha = alpha
        self.K = n_volume_bins
        self.normalizers = normalizers or {"fundamental": mean}
        self._base_seed = seed

        # PettingZoo bookkeeping
        self.possible_agents = ["liq_0", "liq_1"]  # type: ignore[misc]
        self.agents: list[str] = []
        self.timesteps: int = 0

        # ---- spaces ----
        obs_dim = 5
        self.observation_spaces: dict[str, spaces.Box] = {
            aid: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float64,
            )
            for aid in self.possible_agents
        }
        self.action_spaces: dict[str, spaces.Discrete] = {
            aid: spaces.Discrete(self.K) for aid in self.possible_agents
        }

    # ==================================================================
    # PettingZoo ParallelEnv interface
    # ==================================================================

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        if seed is not None:
            self._base_seed = seed
        self._rng = random.Random(self._base_seed)
        np.random.seed(self._base_seed)

        self.agents = self.possible_agents[:]
        self.timesteps = 0

        self._build_market()

        warmup = max(1, int(0.1 * self.n_steps))
        self._warmup_steps = warmup
        self._run_warmup(warmup)
        self._time_offset = warmup

        obs = {aid: self._get_obs(aid) for aid in self.agents}
        infos: dict[str, dict[str, object]] = {aid: {} for aid in self.agents}
        return obs, infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        if self.timesteps >= self.n_steps:
            return self._terminal_step()

        # 1. Both liquidators place orders (coin toss for order)
        order_of_agents = list(self.agents)
        self._rng.shuffle(order_of_agents)

        for aid in order_of_agents:
            if aid not in actions:
                continue
            liq = self._liquidators[aid]
            vol = self._action_to_volume(aid, actions[aid])
            self._place_liquidator_order(liq, vol)

        # 2. Background traders arrive this step
        self._bg_step(self._time_offset + self.timesteps)

        # 3. Market clearing
        self.market.step()
        self._process_matches()

        # 4. Compute rewards
        rewards: dict[str, float] = {}
        for aid in self.agents:
            liq = self._liquidators[aid]
            vol = self._action_to_volume(aid, actions.get(aid, 0))
            rewards[aid] = liq.cash - liq._prev_cash - self.alpha * vol**2
            liq._prev_cash = liq.cash

        # 5. Advance time
        self.timesteps += 1

        # 6. Check truncation (horizon exhausted)
        truncated = dict.fromkeys(self.agents, self.timesteps >= self.n_steps)
        terminated = dict.fromkeys(self.agents, False)

        # If truncated, force-liquidate remaining inventory at final fundamental
        if any(truncated.values()):
            self._force_liquidate()
            for aid in self.agents:
                liq = self._liquidators[aid]
                rewards[aid] += liq.cash - liq._prev_cash
                liq._prev_cash = liq.cash
                terminated[aid] = True

        obs = {aid: self._get_obs(aid) for aid in self.agents}
        infos: dict[str, dict[str, object]] = {aid: {} for aid in self.agents}
        return obs, rewards, terminated, truncated, infos

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _build_market(self) -> None:
        self.fundamental = LazyGaussianMeanReverting(
            mean=self.mean,
            final_time=self.n_steps + 1,
            r=self.r,
            shock_var=self.shock_var,
        )
        self.market = Market(fundamental=self.fundamental, time_steps=self.n_steps)

        # Liquidators (agent IDs: n_bg and n_bg+1)
        base_id = self.n_bg
        self._liquidators: dict[str, LiquidatorAgent] = {}
        self._liq_ids: dict[str, int] = {}
        for i, aid in enumerate(self.possible_agents):
            liq = LiquidatorAgent(
                agent_id=base_id + i,
                market=self.market,
                q_0=self.q_0,
                order_size=self.q_0,
            )
            self._liquidators[aid] = liq
            self._liq_ids[aid] = base_id + i

        # Background ZI agents
        self._bg_agents: dict[int, ZIAgent] = {}
        self._bg_arrivals: dict[int, list[int]] = defaultdict(list)
        n_samples = 10_000
        arrival_times = _sample_arrivals(self.lam_bg, n_samples)
        idx = 0
        for bg_id in range(self.n_bg):
            t = int(arrival_times[idx].item())
            self._bg_arrivals[t].append(bg_id)
            idx += 1
            self._bg_agents[bg_id] = ZIAgent(
                agent_id=bg_id,
                market=self.market,
                q_max=self.q_max,
                shade=self.shade,
                pv_var=self.pv_var,
                est_var=1e6,
            )
        self._bg_arrival_idx = idx
        self._bg_arrival_times = arrival_times
        self._bg_arrival_n_samples = n_samples
        self._match_idx = 0

    def _run_warmup(self, warmup: int) -> None:
        for t in range(warmup):
            self.market.event_queue.set_time(t)
            self._bg_step(t)
            self.market.step()
            self._process_matches_bg()
        # Set market time past warmup
        self.market.event_queue.set_time(warmup)

    def _bg_step(self, t: int) -> None:
        agents_at_t = self._bg_arrivals.get(t, [])
        if agents_at_t:
            self.market.event_queue.set_time(t)
            for aid in agents_at_t:
                agent = self._bg_agents[aid]
                self.market.withdraw_all(aid)
                side = self._rng.choice([BUY, SELL])
                orders = agent.take_action(side)
                self.market.add_orders(orders)

            # Schedule next arrivals for these agents
            if self._bg_arrival_idx >= self._bg_arrival_n_samples - len(self._bg_agents):
                self._bg_arrival_times = _sample_arrivals(self.lam_bg, self._bg_arrival_n_samples)
                self._bg_arrival_idx = 0
            for aid in agents_at_t:
                gap = int(self._bg_arrival_times[self._bg_arrival_idx].item())
                self._bg_arrival_idx += 1
                next_t = t + 1 + gap
                self._bg_arrivals[next_t].append(aid)

    def _place_liquidator_order(self, liq: LiquidatorAgent, volume: float) -> None:
        t = self._time_offset + self.timesteps
        self.market.event_queue.set_time(t)
        self.market.withdraw_all(liq.get_id())
        orders = liq.take_action(volume=volume)
        self.market.add_orders(orders)

    def _process_matches(self) -> None:
        matched = self.market.matched_orders
        n = len(matched)
        for i in range(self._match_idx, n):
            mo = matched[i]
            aid = mo.order.agent_id
            if aid in self._bg_agents:
                q = int(mo.order.order_type * mo.order.quantity)
                c = -mo.price * mo.order.quantity * mo.order.order_type
                self._bg_agents[aid].update_position(q, c)
            else:
                q = mo.order.order_type * mo.order.quantity
                c = -mo.price * mo.order.quantity * mo.order.order_type
                for _liq_aid, liq in self._liquidators.items():
                    if liq.get_id() == aid:
                        liq.update_position(q, c)
                        break
        self._match_idx = n

    def _process_matches_bg(self) -> None:
        matched = self.market.matched_orders
        n = len(matched)
        for i in range(self._match_idx, n):
            mo = matched[i]
            aid = mo.order.agent_id
            if aid in self._bg_agents:
                q = int(mo.order.order_type * mo.order.quantity)
                c = -mo.price * mo.order.quantity * mo.order.order_type
                self._bg_agents[aid].update_position(q, c)
        self._match_idx = n

    def _force_liquidate(self) -> None:
        final_fund = self.market.get_final_fundamental()
        for aid in self.agents:
            liq = self._liquidators[aid]
            remaining = liq.remaining_inventory
            if remaining > 0:
                liq.cash += remaining * final_fund
                liq.position -= remaining

    def _action_to_volume(self, aid: str, action: int) -> float:
        liq = self._liquidators[aid]
        remaining = liq.remaining_inventory
        if self.timesteps >= self.n_steps - 1:
            return remaining
        if self.K <= 1:
            return remaining
        frac = action / (self.K - 1)
        return frac * remaining

    def _get_obs(self, aid: str) -> np.ndarray:
        liq = self._liquidators[aid]
        t = self.timesteps
        time_left = (self.n_steps - t) / self.n_steps
        inv_frac = liq.remaining_inventory / self.q_0
        abs_t = self._time_offset + t
        fund = self.market.fundamental.get_value_at(abs_t)
        best_bid = self.market.order_book.get_best_bid()
        best_ask = self.market.order_book.get_best_ask()
        norm = self.normalizers["fundamental"]

        fund_n = fund / norm
        bid_n = best_bid / norm if not math.isinf(best_bid) else 0.0
        ask_n = best_ask / norm if not math.isinf(best_ask) else fund_n + 0.001

        return np.array(
            [time_left, inv_frac, fund_n, bid_n, ask_n],
            dtype=np.float64,
        )

    def _terminal_step(self):
        obs = {aid: self._get_obs(aid) for aid in self.agents}
        rewards = dict.fromkeys(self.agents, 0.0)
        terminated = dict.fromkeys(self.agents, True)
        truncated = dict.fromkeys(self.agents, True)
        infos = {aid: {} for aid in self.agents}
        self.agents = []
        return obs, rewards, terminated, truncated, infos

    # ==================================================================
    # PettingZoo properties
    # ==================================================================

    @property
    def observation_space(self) -> spaces.Box:
        return self.observation_spaces[self.possible_agents[0]]

    @property
    def action_space(self) -> spaces.Discrete:
        return self.action_spaces[self.possible_agents[0]]
