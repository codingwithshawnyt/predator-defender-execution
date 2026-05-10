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
