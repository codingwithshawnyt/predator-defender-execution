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
