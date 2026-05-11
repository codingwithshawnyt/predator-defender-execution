# Decisions log

Running list of design decisions, in chronological order.

## 2026-05-09: Repo founded

- **Decision:** New repo `predator-defender-execution`. Prior cross-asset OFI work shelved.
- **Rationale:** Pivoted to pure-simulation adversarial RL after determining paid LOB data was infeasible and that the Wellman-group / Lillo-Macri line of work has a clear gap (asymmetric predator-defender + realistic LOB + regime-conditional EGTA).
- **Alternatives considered:** crypto cross-venue OFI on free Binance/Coinbase data; IEX HIST DEEP+ for equities; paid NASDAQ historical.

## 2026-05-09: Simulator choice

- **Decision:** PyMarketSim (umichsrg/pymarketsim) as primary simulator.
- **Rationale:** Wellman group's tool, used in ICAIF 2024 best paper; designed for RL training; closest infrastructure to the literature we extend.
- **Alternatives considered:** ABIDES-Gym (more mature, heavier); ABIDES-MARL (newer, ETH Zurich, less proven).

## 2026-05-09: RL stack

- **Decision:** PPO via Stable-Baselines3 + PettingZoo for multi-agent.
- **Rationale:** PPO more stable than DQN for continuous actions; SB3 well-maintained; PettingZoo handles multi-agent cleanly.
- **Alternatives considered:** RLlib (heavier dependency footprint); DDQL (Lillo-Macri's choice — limited action space).

## 2026-05-09: 18-cell regime grid

- **Decision:** 3 (volatility) × 2 (spread) × 3 (defender size) = 18 cells.
- **Rationale:** Fine enough to characterize regime variation, coarse enough for 100 episodes per cell.
- **Alternatives considered:** Adding time-of-day or message-rate axis (rejected — increases cell count beyond compute budget).

## 2026-05-09: Two-machine workflow via SFTP

- **Decision:** Code authored on Windows dev machine, SFTP'd to remote Linux training workstation. Git origin (GitHub) is the persistent backup.
- **Rationale:** User preference for code-on-dev separation; smaller cognitive overhead than Remote-SSH or pure-git workflows.
- **Alternatives considered:** VSCode Remote-SSH (single source of truth); pure git push/pull (commit friction); rsync (similar to SFTP, less GUI integration).

## 2026-05-10: PyMarketSim API conflicts with methodology

After reading the full PyMarketSim source (wrappers, agents, market, fourheap, fundamental, metrics), six conflicts/gaps surfaced between the simulator's actual API and the assumptions in `docs/methodology.md`:

### Conflict 1: No defender / predator role abstraction

- **Methodology assumes:** Defender liquidating size Q over T=1800s; K predators detecting order-flow signatures.
- **PyMarketSim reality:** Existing agents are ZI (noise), MM (market maker), SpoofingAgent (real-sell + spoof-buy), HBLAgent (belief-learning). No defender that executes a parent order. No predator that detects counterparties from order flow.
- **Resolution needed:** Build `DefenderAgent` (parent-order execution) and `PredatorAgent` (order-flow detection + front-running) from scratch. The `SpoofingAgent` is closest to a predator but its action space (place spoof + real order) doesn't match.

### Conflict 2: Time horizon and step semantics

- **Methodology assumes:** T=1800 seconds, control interval 1s, N=1800 steps.
- **PyMarketSim reality:** Time is discrete integer steps. Gym `step()` advances variable numbers of internal timesteps between self-agent arrivals (via `run_until_next_*_arrival()`). Effective control frequency depends on arrival rate `lam`.
- **Resolution needed:** Define a time-unit convention mapping real seconds to sim steps. Defender arrival rate must be high enough (or fixed-interval) for 1s control.

### Conflict 3: Observation space mismatch

- **Methodology assumes:** Defender knows (inventory, time, LOB state). Predator knows (own state, time, LOB state).
- **PyMarketSim reality:** MMEnv provides 5-dim obs (time_left, fundamental, best_ask, best_bid, inventory) — ignoring the 5 extra metrics it computes. SPEnv adds private values. Neither includes order-flow features needed by the predator.
- **Resolution needed:** Define new obs spaces for both agents. Predator needs order-flow features (signed volume, trade arrival rate, recent matched-order patterns) not in current `metrics.py`.

### Conflict 4: Action space mismatch

- **Methodology assumes:** Defender controls execution schedule (volume per interval). Predator controls front-running aggressiveness.
- **PyMarketSim reality:** MMEnv action is 2-dim beta params. SPEnv action is 2-dim (real price, spoof price). No volume-schedule or front-running-intensity action space exists.
- **Resolution needed:** Design new action spaces. Defender: (volume_to_execute, price_offset). Predator: (detection_threshold, front_run_size, price_offset).

### Conflict 5: No multi-agent Gym interface

- **Methodology assumes:** Independent PPO with adversarial alternation for K+1 agents.
- **PyMarketSim reality:** Each wrapper is single-agent (one RL self agent). MMSPEnv runs MM as a non-RL background agent. No PettingZoo `ParallelEnv` or `AECEnv`.
- **Resolution needed:** Build a PettingZoo ParallelEnv wrapper that manages both defender and predator as simultaneous RL agents.

### Conflict 6: Fundamental variant inconsistency

- **Observation:** MMSPEnv uses eager `GaussianMeanReverting`; MM/SP wrappers use lazy `LazyGaussianMeanReverting`. Both produce the same process.
- **Resolution:** Standardise on `LazyGaussianMeanReverting` for all new environments (memory-efficient for long horizons).

## 2026-05-10: PyMarketSim vendor bug fixes (round 2)

### Fix 7: HBLAgent `Order.__eq__` crash on `peek_order() != None`

- **Bug:** `hbl_agent.py:575` used `self.market.order_book.buy_unmatched.peek_order() != None`. When the heap is empty, `peek_order()` returns `None`, triggering `Order.__eq__(None)` which raises `TypeError`.
- **Fix:** Changed `!= None` → `is not None` for both `buy_unmatched` and `sell_unmatched` checks.
- **Same-class bug in MMSP_wrapper:** `spoofer_arrivals != None` → `is not None`.

### Fix 8: SPEnv `run_agents_only` breaks arrival time tracking

- **Bug:** `SP_wrapper.py:run_agents_only` loop incremented `self.time += 1` inside the loop, but `agents_step()` uses `self.arrivals[self.time]`. When `self.arrivals[t]` was empty, `self.time` still incremented — causing `agents_step` to miss agents at the next time step. Also, only 1 SP arrival was pre-sampled; if it fell within the warmup window (first 10% of sim_time), `run_until_next_SP_arrival` would find no future SP arrivals, raising `ValueError("An episode without spoofer")`.
- **Fix:** (1) Set `self.time = t` at each loop iteration instead of incrementing. (2) Add `self.warmup_steps` attribute. (3) Schedule first SP arrival at `warmup_steps + offset` instead of `offset` alone, ensuring the spoofer always arrives after warmup. Same fix applied to `reset_arrivals`.

### Fix 9: MMSPEnv and MMEnv `run_agents_only` same time-tracking bug

- **Bug:** Both wrappers had the same `self.time += 1` bug in `run_agents_only`. MMSPEnv worked around the SP-arrival-during-warmup issue by hardcoding `+ 1000` offset on SP arrivals, but the general time-tracking bug remained.
- **Fix:** Same pattern as SPEnv fix: set `self.time = t` per iteration, set `self.time = warmup` after loop.

### Fix 10: `metrics.py` invalid escape sequence

- **Bug:** Docstring at line 40 contained `\sum` (LaTeX), which Python 3.12 flags as `SyntaxWarning: invalid escape sequence '\s'`.
- **Fix:** Escaped backslash: `\\sum`.

### Fix 11: MMEnv `normalizers` missing `"reward"` key

- **Bug:** `MM_wrapper.py:run_until_next_MM_arrival` and `end_sim` both divide reward by `self.normalizers["reward"]`, but the example code and docs only provided `fundamental`, `invt`, `cash` keys — causing `KeyError`.
- **Fix:** Not a code change — the normalizers dict must include `"reward"` key. Documented here as a usage requirement.

## 2026-05-10: Time-unit convention for Lillo-Macrì smoke replicate

- **Decision:** 1 sim step = 1 second. Defender/liquidator arrival rate `lam = 1.0` (arrives every step). Episode horizon = N steps (e.g., N = 100 for smoke test, N = 1800 for full runs).
- **Rationale:** Lillo-Macrì use discrete time steps t = 1,...,N with one action per step. PyMarketSim's `run_agents_only` / `run_until_next_arrival` loop advances `self.time` by 1 each iteration. Setting `lam = 1.0` (geometric arrival with p = 1) means the RL agent arrives at every time step — matching Lillo-Macrì's one-action-per-step design. Background ZI traders use lower `lam` (e.g., 0.1) so they arrive stochastically.
- **Impact on observation space:** `time_left = N - t` maps directly to Lillo-Macrì's remaining-steps observation. Inventory `q_t` is in units of shares. Mid-price `S_t` comes from the LOB's `order_book.get_best_bid()/get_best_ask()`.
- **Impact on action space:** Each liquidator's action = volume to sell at this step (continuous float in [0, remaining_inventory]). Mapped to a market sell order at the best bid price (or aggressive cross-spread order for immediate execution).
- **Warmup:** `warmup_steps = max(1, int(0.1 * N))` background-only steps before liquidators enter — consistent with existing SP/MM wrapper convention.

## 2026-05-10: Smoke replicate architecture — SB3 DQN with discretised volume grid

- **Decision:** Use SB3 DQN with a discrete volume grid rather than Lillo-Macrì's custom DDQL with continuous action-as-input. PettingZoo ParallelEnv with two identical `liq_0`, `liq_1` agents.
- **Rationale:** Lillo-Macrì's DDQL uses Q(s, a) where `a` is fed as an input feature to the Q-network — a non-standard architecture not supported by SB3's DQN. SB3 DQN requires a discrete action space. Discretising volume into K bins (e.g., K = 21: 0%, 5%, 10%, ..., 100% of remaining inventory) is a faithful approximation that SB3 can train out of the box. This is a *smoke replicate* — the goal is to verify the tacit-collusion signal, not to exactly reproduce their DDQL architecture.
- **Volume grid:** `action ∈ {0, 1, ..., K-1}` maps to `volume = (action / (K-1)) * remaining_inventory`. K = 21 (5% increments). At t = N (last step), force full liquidation regardless of action.
- **Observation per agent:** 5-dim = `[time_left/N, remaining_inventory/q_0, fundamental/normalizer, best_bid/normalizer, best_ask/normalizer]`. Matches Lillo-Macrì's (t, q_t, S_{t-1}) with LOB bookends added for PyMarketSim realism.
- **Reward per agent per step:** `r_t = execution_price * volume_executed - alpha * volume^2` (Lillo-Macrì Eq. 23, translated to LOB execution). We approximate `alpha` (temporary impact penalty) implicitly through LOB mechanics — selling larger volume moves through the book, yielding worse average price. We also add an explicit quadratic penalty `alpha * v_t^2` to match their formulation and ensure the IS vs volume tradeoff is learnable.
- **Success criterion:** Cumulative reward converges toward symmetric TWAP-Pareto level (not closed-form Nash). The signature is supra-competitive IS: both agents' average IS between Nash and Pareto-optimal, with centroid clustering near the Pareto front.
- **Training:** DQN with default SB3 hyperparameters (MlpPolicy, lr=1e-4, batch_size=64, buffer_size=15000, gamma=1.0, epsilon greedy). 5000 episodes for smoke test. Both agents trained with independent DQN instances via PettingZoo ParallelEnv + SB3 per-agent training loop.
- **Alternatives considered:** (1) Custom DDQL exactly replicating Lillo-Macrì — too much engineering for a smoke test. (2) PPO with continuous action — valid but different algorithmic family; DQN closer to their DQN-family approach. (3) Single-agent SB3 with opponent frozen — doesn't test simultaneous learning; we need both agents learning.

## 2026-05-10: LiquidatorAgent as PyMarketSim agent

- **Decision:** `LiquidatorAgent` implements `marketsim.agent.agent.Agent` (the vendor ABC). It is a *shell* agent — the RL policy selects the volume, and `LiquidatorAgent.take_action(volume)` translates that volume into a sell `Order` at the best bid price (crossing the spread for immediate execution).
- **Rationale:** PyMarketSim's `Market.add_orders()` expects `List[Order]`. The `Agent` ABC requires `take_action() -> List[Order]`. Our `LiquidatorAgent` bridges the RL action (volume) to the market's order interface. The RL wrapper calls `liquidator.take_action(volume=action)` instead of the default no-arg `take_action()`.
- **Order placement strategy:** Sell `volume` shares at `price = best_bid` (aggressive, crosses spread). This approximates Lillo-Macrì's Almgren-Chriss reduced-form execution (which always executes at the impacted price) in an LOB context. If `volume` exceeds available bid depth, the order rests in the book as a limit order and may partially fill in subsequent steps.
- **Inventory tracking:** `LiquidatorAgent.position` tracks cumulative signed quantity; `LiquidatorAgent.cash` tracks cumulative cash from executions. `remaining_inventory = q_0 - position` (where position is negative for a seller).
- **Future extension:** `DefenderAgent(LiquidatorAgent)` will add defense-specific features (randomization, decoy orders). `PredatorAgent` will share the same `Agent` ABC but with a different `take_action()` signature (buy-side + front-running).

## 2026-05-10: Vendor bug fixes (round 3) — smoke replicate end-to-end

### Fix 12: Time-offset bug — liquidator/BG orders scheduled at wrong event_queue time

- **Bug:** After warmup, the event queue's `current_time` was at `warmup_steps` (e.g., 10). But `env.step()` called `set_time(self.timesteps)` (starting at 0), which rewound the event queue to time 0. Liquidator orders were then scheduled at `scheduled_activities[0]`, but warmup activities at time 0 had already been consumed. When `market.step()` processed time 0, it would re-process stale warmup orders or miss liquidator orders entirely. In some cases, the liquidator orders were never matched because they were queued at a time step that had already passed.
- **Fix:** Introduced `self._time_offset = warmup_steps`. All calls to `event_queue.set_time()` and fundamental value lookups now use `self._time_offset + self.timesteps` (absolute market time). This ensures liquidator and BG orders are scheduled at the correct time slot after warmup, and `market.step()` processes them as expected.
- **Impact:** Without this fix, liquidators almost never got fills (their orders were lost in the event queue). After the fix, both liquidators receive fills consistently.

### Fix 13: `_process_matches` iterated all matched_orders cumulatively

- **Bug:** `_process_matches()` iterated `self.market.matched_orders` from index 0 every call. This list grows monotonically across the episode (old matches are never removed). This meant warmup matches were re-processed on every step, causing BG agent positions to be double/triple-counted.
- **Fix:** Track `self._match_idx` (last processed index). On each call, only process matches from `_match_idx` to `len(matched_orders)`, then update `_match_idx`. Same fix applied to `_process_matches_bg()`.
- **Impact:** Prevented position/cash corruption for BG agents, which was causing `PrivateValues.value_for_exchange` IndexError when BG positions became non-integer floats.

### Fix 14: BG agent `update_position` receives float → `PrivateValues` IndexError

- **Bug:** `ZIAgent.update_position(q, c)` increments `self.position` by `q`. The `q` was computed as `mo.order.order_type * mo.order.quantity`, which is a `float` when liquidator order quantities are fractional. `PrivateValues.value_for_exchange()` uses `position` as a list index, which requires `int`. A `float` position caused `IndexError: only integers... are valid indices`.
- **Fix:** Cast `q` to `int` for BG agents in `_process_matches()` and `_process_matches_bg()`. Liquidator agents keep float positions (they don't use PrivateValues).

### Fix 15: SB3 DQN `env=None` crashes at `_setup_model()`

- **Bug:** DQN was initialized with `env=None` then had `observation_space`/`action_space`/`n_actions` set manually before calling `_setup_model()`. But `DQN.__init__` itself calls `_setup_model()` → `set_random_seed()` → `action_space.seed()`, which fails because `action_space` doesn't exist yet on the DQN object.
- **Fix:** Create a minimal `_SingleAgentEnv(gym.Env)` subclass that provides the correct observation/action spaces, and pass an instance to `DQN.__init__`. SB3 uses the env only for space introspection; the training loop still uses the PettingZoo env directly.

### Fix 16: Training loop used `env.agents` post-step (may be empty)

- **Bug:** After `env.step()`, if all agents are terminated/truncated, `env.agents` becomes `[]`. The replay buffer storage loop iterated `env.agents`, causing `KeyError` when trying to access `actions[aid]`.
- **Fix:** Capture `active_agents = list(env.agents)` before the step, and use `active_agents` for both action selection and replay buffer storage.
