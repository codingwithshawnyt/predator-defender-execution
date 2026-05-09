# Literature review

## Three clusters and the seam between them

The literature on adversarial execution clusters into three groups, with a specific gap at their intersection that this paper targets.

### Cluster 1: Classical analytical predatory trading

Brunnermeier & Pedersen 2005 (BP), Carlin-Lobo-Viswanathan 2007, Schoneborn-Schied 2009, Schied-Zhang 2017/2019, Carmona-Yang 2011, Micheli-Muhle-Karbe-Neuman 2023.

These derive closed-form predator-prey equilibria in Almgren-Chriss-style reduced-form market impact models. In BP 2005: one trader is forced to liquidate, predators sell first and buy back, prey overshoots. Linear permanent + temporary impact, no LOB, no learning, no microstructure regime structure beyond volatility entering as a parameter.

**Doesn't do:** learned policies via deep RL, realistic LOB simulator, regime-conditional analysis.

### Cluster 2: RL for single-agent execution

Ning-Lin-Jaimungal 2021, Macri-Lillo 2024, Hafsi-Vittori 2024 (ICAIF), Capponi-Menkveld-Zhang 2024.

Single-agent RL liquidation against a non-strategic environment. Strong methodological progress on actor-critic, online execution, time-varying liquidity. No adversarial agent.

**Doesn't do:** game-theoretic equilibrium, predator role.

### Cluster 3: Multi-agent equilibrium learning in markets

- **Lillo & Macri 2024 (arXiv:2408.11773)** — two **symmetric** liquidators with DDQL in Almgren-Chriss; tacit collusion emerges, converges near Pareto-optimal TWAP rather than Nash. Tests volatility transfer (train low, test high). They explicitly suggest multi-asset/multi-agent extensions and time-varying liquidity as future work.
- **Cheridito-Dupret-Wu 2025** (ABIDES-MARL) — extends ABIDES-Gym for synchronized multi-agent learning. Demonstrated on Kyle-model price discovery and a liquidity-trader problem in a realistic LOB. Symmetric framing, no predator-prey asymmetry.
- **Gu-Wang-Mascioli-Chakraborty-Wellman 2024** (ICAIF best paper) — spoofing/manipulation under EGTA in PyMarketSim. Manipulator + market participants. Not execution-defense.
- **Wellman-Tuyls-Greenwald 2024** (JAIR) — EGTA survey; EGTA has been applied to spoofing, latency arbitrage, prediction markets, but **not to predator-defender execution in realistic LOBs**.

## The seam

No paper combines (a) asymmetric predator-defender roles, (b) learned policies via deep RL, (c) realistic LOB simulator with microstructure detail, (d) regime-conditional empirical equilibrium analysis. That is the niche this paper occupies.

## Direct extensions of Lillo-Macri 2024

We extend their work in three specific ways:

1. **Asymmetric roles instead of symmetric.** They have two liquidators both wanting to sell. We have a defender liquidating and predators choosing to predate — closer to BP 2005's economic setup.
2. **Realistic LOB instead of Almgren-Chriss reduced-form.** They use closed-form market impact. We use PyMarketSim with discrete order book mechanics, message-level events, market makers.
3. **Regime axes beyond volatility.** They test volatility transfer between training and testing. We add spread regime and defender-size regime, and report the empirical equilibrium per cell instead of just transfer behavior.

## Risks

- **Wellman group might publish first.** They have the infrastructure and methodology. Mitigation: move fast, lean into the regime-conditional angle they've shown less interest in.
- **Lillo-Bologna group might extend to predator-prey.** Their direction has been tacit collusion in symmetric games. If they do publish predator-prey, our realistic-LOB + EGTA framing differentiates.
- **Wellman-group spoofing line might encroach.** Our defense focus (rather than detection of manipulation) is the orthogonal angle.
