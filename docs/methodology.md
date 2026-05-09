# Methodology — Predator-Defender Execution Equilibria

**Status:** v0.1, locked prior to repo creation. All design choices below are committed unless explicitly revisited via PR.

## Working title

Regime-Dependent Predator-Defender Equilibria in Realistic Limit Order Books: A Multi-Agent Reinforcement Learning Study.

## Core research question

When a defender executes a parent order in a realistic LOB while one or more predators learn to detect and front-run, what is the empirical equilibrium structure, and how does it shift across liquidity regimes?

## Three locked claims

If a claim cannot be supported on the simulated data after honest analysis, the paper ships the negative result for that claim — not nothing.

1. **C1 (Asymmetric equilibrium recovery).** When the predator's detection ability is good, the empirical equilibrium recovers the qualitative Brunnermeier-Pedersen 2005 result: predators sell first, prey overshoots, defender pays elevated implementation shortfall vs. the no-predator baseline. Quantitative gap reported.

2. **C2 (Regime dependence).** The equilibrium structure varies non-trivially across regimes. In high-volatility regimes, predator detection deteriorates, pushing the equilibrium back toward the no-predator baseline. In low-spread / high-depth regimes, predation profitability rises. The functional form of the equilibrium-vs-regime mapping is the empirical contribution.

3. **C3 (Defense effectiveness).** Schedule randomization, decoy orders, and adaptive (regime-conditional) execution policies measurably reduce defender regret-to-no-predator vs. fixed schedules (TWAP, Almgren-Chriss). Dose-response curve (randomization intensity vs. regret reduction vs. tracking-error cost) reported.

Figure mapping: C1 → equilibrium phase diagram; C2 → regret-by-regime heatmap; C3 → defense Pareto frontier.

## Game setup

**Roles.** One defender liquidating size Q over horizon T = 1800 seconds with control interval 1 s (N = 1800 steps). K predators with K ∈ {1, 3}. Background population: zero-intelligence noise traders + momentum traders + market makers, calibrated to roughly match stylized facts.

**Information structure.** Defender knows: own inventory, time, public LOB state. Predators know: own state, time, public LOB state — but not the defender's identity or schedule directly. Predators must learn to detect the defender from order flow signatures.

## Regime axes (3 axes, 18 cells)

- **Volatility regime** (3 levels: low / mid / high) — controlled by background-trader fundamental-volatility parameter
- **Spread regime** (2 levels: tight / wide) — controlled by market-maker quoting policy
- **Defender-size regime** (3 levels: small / medium / large Q relative to typical horizon ADV)

Headline reports marginals + 2-way interactions (vol × size, spread × size).

## Training procedure

Per regime, train defender and predators jointly via independent PPO with adversarial alternation. Burn-in: predators frozen at naive front-runner while defender pre-trains. Then alternating updates. Verify convergence via best-response gap below 5% of no-predator baseline IS.

## EGTA reduction

After RL training, distill each agent's policy into a discrete archetype set and build empirical payoff matrices per regime.

- **Defender archetypes (5):** TWAP, Almgren-Chriss, randomized-TWAP, RL-trained-against-no-predator, RL-trained-adversarially
- **Predator archetypes (5):** no-action, naive-front-runner, threshold-OFI-detector, RL-trained, RL-trained-adversarially

Compute symmetric Nash among predators and mixed-strategy equilibrium across roles via `gameanalysis`.

## Baselines

**Defender:** B1 TWAP / B2 Almgren-Chriss / B3 Schied-Zhang Nash / B4 RL-vs-no-predator / B5 RL-vs-naive-predator / B6 multi-agent adversarial RL (proposed).

**Predator:** P1 no predator / P2 BP-2005-vs-TWAP / P3 OFI-threshold detector / P4 RL-trained (proposed).

## Evaluation protocol

- Test seeds disjoint from training seeds
- Per regime: 100 episodes per (defender, predator) policy pair
- Bootstrap CIs over episode seeds (block-bootstrap if temporal dependence)
- Diebold-Mariano for IS comparisons across defender policies
- Holm-Bonferroni correction across regimes
- 5 training seeds for headline numbers

## Detection signature analysis (for C3)

Per defender policy: lag-h cross-correlation, PSD of defender's order arrivals, mutual information between observable LOB statistics and hidden inventory.

## Statistical-rigor checklist

- [ ] Test seeds disjoint from training
- [ ] Multiple training seeds (≥ 5) for headline numbers
- [ ] Best-response gap reported for convergence claim
- [ ] Block bootstrap CIs on IS and predator profit
- [ ] Diebold-Mariano with Holm-Bonferroni
- [ ] Negative-result regime explicitly reported
- [ ] All hyperparameters in version-controlled config files

## Limitations

- Findings are about a *calibrated simulator*, not real markets; generalizability to live trading is bounded
- PyMarketSim is single-asset in current configuration; multi-asset extension is future work
- Detection signature analysis is in our simulator's coordinates; real-market signatures differ
- 18 regime cells is coarse; finer-grained analysis is appendix-only with cell-count thresholds
