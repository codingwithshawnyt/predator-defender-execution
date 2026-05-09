# predator-defender-execution

Regime-dependent equilibria in predator-defender execution games via multi-agent reinforcement learning and empirical game-theoretic analysis (EGTA) in realistic limit order book simulators.

## Research question

When a defender executes a parent order in a realistic LOB while one or more predators learn to detect and front-run, what is the empirical equilibrium structure, and how does it shift across liquidity regimes?

See `docs/methodology.md` for the full pre-registered methodology and `docs/literature_review.md` for the literature context.

## Setup

```powershell
# Install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone and enter
git clone git@github.com:codingwithshawnyt/predator-defender-execution.git
cd predator-defender-execution

# Sync dependencies
uv sync --extra dev

# Activate
.\.venv\Scripts\Activate.ps1

# Set up pre-commit
pre-commit install

# Smoke test
uv run pytest tests/unit -v
```

For Linux/macOS the install is `curl -LsSf https://astral.sh/uv/install.sh | sh` and the venv activation is `source .venv/bin/activate`. Everything else is the same.

## Common commands

```powershell
uv run ruff check src tests           # lint
uv run ruff format src tests          # format
uv run mypy src                       # type check
uv run pytest tests/unit -v           # unit tests
uv run pytest tests -v                # all tests
uv run python -m pde.training.smoke_replicate    # Phase-1 smoke replication
```

## Layout

```
src/pde/
├── sim/        # PyMarketSim wrappers, environment construction
├── agents/     # Defender + predator policies (baselines and RL)
├── training/   # PPO training loops, adversarial alternation
├── egta/       # Empirical game-theoretic analysis: payoff matrices, Nash solver
├── regimes/    # Regime axis definitions and sampling
├── eval/       # IS, regret, profit metrics; statistical tests
├── figures/    # One script per paper figure
└── analysis/   # Detection signature analysis, capacity sweeps

configs/        # Hydra configs (one per experiment)
notebooks/      # Exploratory work, one notebook per claim
docs/           # Methodology, decisions log, references, lit review
tests/          # pytest unit + integration tests
artifacts/      # Runtime outputs (checkpoints, logs, figures) — gitignored
```

## Development workflow

This repo is authored on a Windows dev machine (no GPU training) and synced to a remote Linux workstation for training runs. Synchronization is via the VSCode SFTP plugin; the plugin's local config (`.vscode/sftp.json`) is gitignored. Commit and push to GitHub from the dev machine; on the remote workstation, pull from GitHub for clean releases or accept SFTP'd files for hot iteration.

## License

MIT.
