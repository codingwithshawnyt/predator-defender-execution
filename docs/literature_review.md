# Literature Review: Adversarial Execution, EGTA, and Market Impact Games

---

## Overview: Three Clusters and the Seam Between Them

The literature on adversarial execution clusters into three groups, with a specific gap at their intersection that this paper targets.

### Cluster 1: Classical analytical predatory trading

Brunnermeier & Pedersen 2005 (BP), Carlin-Lobo-Viswanathan 2007 (CLV), Schoneborn-Schied 2009, Schied-Zhang 2017/2019, Carmona-Yang 2011, Micheli-Muhle-Karbe-Neuman 2023.

These derive closed-form predator-prey equilibria in Almgren-Chriss-style reduced-form market impact models. In BP 2005: one trader is forced to liquidate, predators sell first and buy back, prey overshoots. Linear permanent + temporary impact, no LOB, no learning, no microstructure regime structure beyond volatility entering as a parameter.

**Doesn't do:** learned policies via deep RL, realistic LOB simulator, regime-conditional analysis.

### Cluster 2: RL for single-agent execution

Ning-Lin-Jaimungal 2021, Macri-Lillo 2024, Hafsi-Vittori 2024 (ICAIF), Capponi-Menkveld-Zhang 2024.

Single-agent RL liquidation against a non-strategic environment. Strong methodological progress on actor-critic, online execution, time-varying liquidity. No adversarial agent.

**Doesn't do:** game-theoretic equilibrium, predator role.

### Cluster 3: Multi-agent equilibrium learning in markets

- **Lillo & Macri 2024** (arXiv:2408.11773) — two **symmetric** liquidators with DDQL in Almgren-Chriss; tacit collusion emerges, converges near Pareto-optimal TWAP rather than Nash. Tests volatility transfer (train low, test high). They explicitly suggest multi-asset/multi-agent extensions and time-varying liquidity as future work.
- **Cheridito-Dupret-Wu 2025** (ABIDES-MARL) — extends ABIDES-Gym for synchronized multi-agent learning. Demonstrated on Kyle-model price discovery and a liquidity-trader problem in a realistic LOB. Symmetric framing, no predator-prey asymmetry.
- **Gu-Wang-Mascioli-Chakraborty-Wellman 2024** (ICAIF best paper) — spoofing/manipulation under EGTA in PyMarketSim. Manipulator + market participants. Not execution-defense.
- **Wellman-Tuyls-Greenwald 2024** (JAIR) — EGTA survey; EGTA has been applied to spoofing, latency arbitrage, prediction markets, but **not to predator-defender execution in realistic LOBs**.

## The Seam

No paper combines (a) asymmetric predator-defender roles, (b) learned policies via deep RL, (c) realistic LOB simulator with microstructure detail, (d) regime-conditional empirical equilibrium analysis. That is the niche this paper occupies.

## Direct Extensions of Lillo-Macri 2024

We extend their work in three specific ways:

1. **Asymmetric roles instead of symmetric.** They have two liquidators both wanting to sell. We have a defender liquidating and predators choosing to predate — closer to BP 2005's economic setup.
2. **Realistic LOB instead of Almgren-Chriss reduced-form.** They use closed-form market impact. We use PyMarketSim with discrete order book mechanics, message-level events, market makers.
3. **Regime axes beyond volatility.** They test volatility transfer between training and testing. We add spread regime and defender-size regime, and report the empirical equilibrium per cell instead of just transfer behavior.

## Risks

- **Wellman group might publish first.** They have the infrastructure and methodology. Mitigation: move fast, lean into the regime-conditional angle they've shown less interest in.
- **Lillo-Bologna group might extend to predator-prey.** Their direction has been tacit collusion in symmetric games. If they do publish predator-prey, our realistic-LOB + EGTA framing differentiates.
- **Wellman-group spoofing line might encroach.** Our defense focus (rather than detection of manipulation) is the orthogonal angle.

---

## Paper 1: Wellman, Tuyls, Greenwald (2024) — EGTA Survey

**Reference:** "Empirical Game-Theoretic Analysis: A Survey." Journal of Artificial Intelligence Research 82:1017-1076, 2025. arXiv:2403.04018.

**Scope:** 60-page survey covering 20+ years of EGTA methodology. This is the definitive reference for the framework we adopt.

### 1.1 What EGTA Is and Why It Matters

EGTA = inducing a game model from agent-based simulation data, then applying game-theoretic reasoning (equilibrium computation, strategy evaluation) to that induced model. The core idea: "employ agent-based simulation to generate data from which we can induce a game model, which we call the **empirical game**."

**Key distinction from traditional/analytic game theory:**

- Traditional GT requires an **analytic/declarative game model** (tables, trees, explicitly specified utility functions). Feasible only for artificially defined games or highly stylized models.
- EGTA starts from a **procedural description** (simulator) and **induces** the game model empirically. The model "comes not from declarative representation, but is derived by interrogation of a procedural description of the game environment."
- EGTA "decouples descriptive complexity from game-theoretic reasoning complexity" — you can simulate a complicated world to produce an empirical game as simple or complex as you can computationally afford.
- ABM alone is "too flexible" (open-ended outcomes); EGTA inherits from game theory the mathematical framework for rational choice and equilibrium.

**Three key departures of empirical games from the "underlying game":**

1. The simulator itself may only approximate the underlying game.
2. The empirical game covers only a **strict subset** of possible strategies.
3. Payoffs are induced from noisy/sparse simulation data → approximation error.

Notation: Γ̂ = empirical game model; û_i = empirical utility functions, distinguished from the true game Γ and true utilities u_i.

### 1.2 The EGTA Process: Subproblems

The paper organizes EGTA as an **iterative process**:

1. **Simulation**: Run agent-based simulation with selected strategy profiles → generate payoff data
2. **Game model induction** (estimation/learning): Construct/update empirical game model Γ̂ from simulation data
3. **Game analysis** (solving): Compute equilibria or other solutions of Γ̂
4. **Strategy exploration**: Use analysis results to identify new strategies to add to restricted strategy sets X_i
5. **Profile selection**: Determine which strategy profiles to simulate next
6. Loop back to step 1

**Core subproblems:**

- **Heuristic strategy specification** (Section 3.1): Defining restricted strategy sets X_i ⊆ S_i
- **Game model induction** (Section 3.2): Estimating payoffs from simulation data; handling incomplete game models; game model learning (regression/ML)
- **Player reduction** (Section 3.2.3): Scaling to many-player games via aggregation
- **Game solving** (Section 3.3): Computing equilibria on empirical games, using HPT representation, replicator dynamics, subgame search
- **Strategy evaluation** (Section 3.4): Ranking/assessing strategies (NE-regret, Nash-averaging, α-Rank)
- **Statistical reasoning** (Section 4): Variance reduction, confidence bounds, sampling control, bootstrap
- **Strategy exploration** (Section 5): Automated strategy generation, PSRO
- **Empirical mechanism design** (Section 7): Using EGTA to evaluate/optimize mechanisms

### 1.3 Strategy Exploration Methods

**Manual approach**: Design heuristic strategies based on domain knowledge, often parameterized (e.g., bidding shading factor ρ).

**Automated strategy generation** (Section 5.1):

- **Double Oracle (DO)** (McMahan et al., 2003): Start with restricted X_i, compute NE σ* of Γ↓X, then augment: X_i^{k+1} = X_i^k ∪ {BR_i(σ*_{-i})}. Terminate when BR is already in X. This is best-response to the current NE.
- **Genetic optimization**: Phelps et al. (2006) used genetic search over parametric strategy space, optimizing performance against the equilibrium of the empirical game.
- **RL as best-response oracle**: Schvartzman & Wellman (2009, 2010) used RL (tile-coded Q-functions) to approximate the BR oracle in continuous double auctions. These were essentially DO instances using RL-approximated BR.
- **Local search**: Used for protocol compliance (Wellman et al., 2013) and credit network formation (Dandekar et al., 2015).

**PSRO** (Section 5.2) — the dominant modern framework (see below).

**Key insight**: The **strategy exploration problem** (Jordan et al., 2010a) = in which order to introduce strategies. Adding strategies can *increase* regret before eventually decreasing it (Table 6 example). No guarantee of monotone progress in general.

**MRCP (Minimum Regret Constrained Profile)**: MRCP(Γ,X) = argmin_{σ∈Δ(X)} ε^Γ(σ). Using MRCP as an MSS ensures monotone improvement (anytime property), but may not always perform best in practice. Wang & Wellman (2023b) propose balancing equilibrium and MRCP via regularization.

### 1.4 Payoff Function Estimation

**Direct estimation** (most straightforward): û_i(s) = sample average over simulation runs of profile s.

**Game model learning / regression** (Section 3.2.2):

- Vorobeychik et al. (2007) — first to apply regression to learn payoff functions from simulation data.
- Sokota et al. (2019) — learn **deviation payoffs** (not raw payoffs), advantageous for equilibrium computation.
- Li & Wellman (2021) — deviation-payoff learning for symmetric Bayesian games.
- Gatchel & Wiedenbeck (2023) — learning models covering **families of games** parameterized by context features.

**Variance reduction** (Section 4.1): Control variates exploit correlation between payoffs and observable variables. Achieved up to 50% simulation reduction in TAC supply chain game.

**Statistical bounds** (Section 4.3): If Γ̂ is an ε-uniform approximation of Γ, then ℰ(Γ) ⊆ ℰ_{2ε}(Γ̂) ⊆ ℰ_{4ε}(Γ). Hoeffding-based and Bennett's inequality bounds available for sample complexity.

### 1.5 Game Reduction Techniques

**Player reduction** (Section 3.2.3) — approximating many-player games by fewer-player games:

1. **Hierarchical reduction** (Wellman et al., 2005b): Each reduced player controls n/p of the full-game players.
2. **Twins reduction** (Ficici et al., 2008): n-player → 2-player. **Preserves symmetric PSNE**: (s,s) is NE of Γ^(2) twins-reduced iff everyone playing s is NE of Γ.
3. **Deviation-Preserving Reduction (DPR)** (Wiedenbeck & Wellman, 2012): Generalizes twins for p > 2. Also **preserves symmetric PSNE**. Brinkman (2018) used DPR (p=4 or 6) to analyze financial market games with up to **216 agents**.

**Symmetry exploitation**: Role-symmetric games reduce profile space from |S|^n to (n+|S|-1 choose n) distinct profiles.

### 1.6 Equilibrium Computation Methods

1. **Replicator Dynamics (RD)**: Iterative improvement on the strategy simplex. Not guaranteed to find solutions; multiple starts recommended. Wiedenbeck & Brinkman (2023) achieved 10^4-fold speedup using HPT data structures.
2. **Subgame search**: Search over maximal complete subgames. Find solutions in subgames, then test deviations.
3. **Minimum-Regret-First Search (MRFS)** (Jordan et al., 2008): Maintains lower bounds on regret for evaluated profiles.
4. **α-Rank** (Omidshafiei et al., 2019): Markov chain-based solution concept. Used for evaluating AlphaZero checkpoints.
5. **Bootstrap for regret estimation** (Wiedenbeck et al., 2014): Resample payoff data to build distribution of regret values → quantify uncertainty.
6. **Gaussian processes** (Al-Dujaili et al., 2018; Picheny et al., 2019): Treat game-solving as black-box optimization; GP regression for probabilistic game model; acquisition functions based on regret.

### 1.7 PSRO: The Key Modern Framework

**PSRO = Policy-Space Response Oracles** (Lanctot et al., 2017). The key framework combining deep RL with EGTA.

Each iteration: (1) compute/update empirical game U^Π over current strategies X^e, (2) derive training targets σ_{-i} via **meta-strategy solver (MSS)**, (3) use deep RL to train a best-response policy to σ_{-i}, (4) add to strategy pool.

**MSS options**: Nash equilibrium (= double oracle), uniform (= fictitious play), projected replicator dynamics, rectified Nash (zero-sum), α-Rank, correlated equilibrium, MRCP, online learning profiles, diversity-regularized targets.

**AlphaStar** used a "Nash league" tracking equilibria of candidate policies.

### 1.8 Applications to Financial Markets

- **Continuous Double Auctions**: Phelps et al. (2005), Schvartzman & Wellman (2009, 2010) — RL-generated CDA strategies.
- **Financial markets broadly**: Brinkman (2018) PhD thesis — "Understanding Financial Market Behavior through EGTA" — used DPR with up to 216 agents.
- **Market manipulation / spoofing**: Wang et al. (2021) — EGTA over 36 instances with/without a market manipulator. Liu et al. (2022) — market manipulation implications.
- **Prediction markets**: Wah et al. (2016).
- **ETF markets**: Shearer et al. (2021).
- **Financial regulation**: Cheng & Wellman (2017).
- **Pursuit-evasion games**: Li et al. (2023a) — but NOT in the financial market context.

**Critical gap**: EGTA has NOT been applied to **predator-defender execution in realistic LOBs**. The financial market applications focus on macro-level trading games, auction mechanisms, and manipulation — not adversarial execution defense.

### 1.9 Key Open Problems

1. **Extending EGTA beyond normal form**: Most EGTA assumes normal-form empirical games. Extending to extensive-form, mean-field, and team games requires new innovations.
2. **Restricted-game vs. base-game solutions**: "We typically lack strong theoretical connections between restricted-game and base-game solutions (except in the asymptotic limit)." Need better theory.
3. **Generalizability beyond specific game instances**: "Our interest is not actually for any particular game instance, but rather in a class of strategic situations."
4. **MSS selection**: "There is no definitive understanding of which MSS is the best to employ for a given game environment."
5. **Non-monotone exploration**: Strategy exploration can increase regret before decreasing it.
6. **Statistical reliability**: Even with uniform approximation bounds, connections between empirical-game solutions and true-game solutions are only probabilistic.

### 1.10 Relevance to Our Work

- **Our EGTA pipeline**: We will use PSRO (or a simplified DO variant) with PyMarketSim as the simulator. Each "strategy" is a trained RL policy for the defender or predator. The empirical game is induced over a grid of regime parameters (volatility, spread, defender size).
- **Strategy exploration**: Start with heuristic strategies (TWAP, VWAP, aggressive front-loading) and RL-learned best responses. Use Nash as the MSS.
- **Player reduction**: If we scale to K > 1 predators, we can use DPR or twins reduction to keep the empirical game manageable.
- **Statistical bounds**: We should report ε-regret and confidence intervals for our empirical equilibria, following Section 4's methodology.
- **Gap we fill**: First EGTA application to **asymmetric predator-defender execution in a realistic LOB**.

---

## Paper 2: Lillo & Macri (2024) — DDQL Market Impact Game

**Reference:** "Deviations from the Nash equilibrium in a two-player optimal execution game with reinforcement learning." arXiv:2408.11773. Revised February 2026.

**Why critical:** This is our **replication target**. We extend it from symmetric to asymmetric roles and from Almgren-Chriss to realistic LOB.

### 2.1 Model Setup

Two risk-neutral agents (λ = 0) liquidating identical initial portfolios of q_0 = 100 shares at S_0 = $10 over N = 10 time steps in the Almgren-Chriss framework.

**Price dynamics (per time step):**

```
S_t = S_{t-1} - κ(V_t/τ)τ + σ√τ ξ_t           (mid-price)
S̃_t^(k) = S_{t-1} - α(v_t^(k)/τ)               (execution price for agent k)
```

where V_t = v_t^(1) + v_t^(2) is total traded volume, ξ_t ~ N(0,1).

**Impact parameters:** α = 0.002 (temporary), κ = 0.001 (permanent).

**Volatility regimes:** σ = 10^{-9} (zero noise), 10^{-3} (moderate), 10^{-2} (large).

### 2.2 Analytical Nash Equilibrium (citing Schied & Zhang 2017)

For two risk-neutral agents with equal inventories, the unique Nash equilibrium is (Eq. 7-8):

```
q_t^(1)* = ½(Σ(t) + Δ(t))
q_t^(2)* = ½(Σ(t) − Δ(t))
```

where:

```
Σ(t) = Q·e^{-κt/(6α)} · sinh((N-t)√(κ²+12αλσ²)/(6α)) / sinh(N√(κ²+12αλσ²)/(6α))
Δ(t) = Q̃·e^{κt/(2α)} · sinh((N-t)√(κ²+4αλσ²)/(2α)) / sinh(N√(κ²+4αλσ²)/(2α))
```

For risk-neutral agents (λ=0): Δ(t) = 0 (since Q̃ = 0 for symmetric inventories), so both agents use the same strategy.

### 2.3 Pareto-Optimal (Collusive) Strategy — Theorem 2

For risk-neutral agents, the **Pareto-optimal strategy is TWAP**: v_t^(1) = v_t^(2) = q_0/N for all t.

Proof: Minimizing F(v^(1,2)) = E[IS(v^(1)|v^(2))] + E[IS(v^(2)|v^(1))] subject to inventory constraints yields v_t = q_0/N via Lagrangian. The first and last terms in the derivative cancel, leaving exactly TWAP.

### 2.4 DDQL Implementation — Complete Hyperparameters

| Parameter | Value |
|-----------|-------|
| NN layers | 5 |
| Hidden nodes per layer | 30 |
| Activation | LeakyReLU |
| Optimizer | ADAM |
| Learning rate | 0.0001 |
| Training episodes (C) | 5,000 |
| Testing episodes (M) | 2,500 |
| Number of independent runs | 20 (main), 10 (variable vol) |
| Batch size (b) | 64 |
| Replay buffer max length (L) | 15,000 |
| Discount factor (γ) | 1 (no discounting) |
| ε initial | 1 (full exploration) |
| ε decay rate (c) | 0.995 |
| ε reset interval (m) | 75 actions |
| Target net update | Every 75 actions (hard copy: Q_tgt ← Q_main) |

**Architecture**: Each agent has TWO networks: Q_main (action selection) and Q_tgt (target computation). Both share the same 5-layer, 30-node architecture. Weights evolve independently.

### 2.5 State and Action Representation

**State tuple**: g_t^i = (t, q_t^i, S_{t-1}) — each agent knows current time, own remaining inventory, and the permanently impacted mid-price. **No information about the other agent's inventory or actions.**

**Q-network input**: (q_{i,t}, t, S_{t-1}, v_{i,t}) — 4-dimensional. The action v_{i,t} is included as an input feature, so the network approximates Q(s, a) for **continuous** actions. This is NOT a standard DQN with a fixed discrete action grid.

**Normalization**: All inputs normalized to [-1, 1].

### 2.6 Action Selection — Continuous, Not Discrete

**During exploration** (probability ε): v_t ~ N(μ = q_t/(N-t), δ = |q_t/(N-t)|) — Gaussian centered on the TWAP-like remaining-inventory rate.

**During exploitation** (probability 1-ε): v_t = argmax_{v' ∈ [0,q_0]} Q_main(g_t, v' | θ_main) — continuous optimization over [0, q_0].

**Constraint**: At t = N (last step), the agent must sell all remaining inventory.

### 2.7 Reward Function

**Per-step reward** (Eq. 23):

```
r_{t,i} = S_{t-1} · v_{t,i} - α · v_{t,i}²
```

**Episode reward** (cumulative, equivalent to negative IS):

```
R_i = -q_0 · S_0 + Σ_{t=1}^{N} [S_{t-1} · v_{t,i} - α · v_{t,i}²]
```

The other agent's actions indirectly impact the reward through S_{t-1} (which incorporates permanent impact from both agents).

### 2.8 Training Procedure

- Both agents trained **simultaneously** with the same ε schedule.
- At each time step, a **coin toss** (u ~ Bernoulli(0.5)) decides which agent trades first. This ensures symmetry.
- ε starts at 1, decays by factor 0.995 every 75 actions.
- Target network: hard copy Q_tgt ← Q_main every 75 actions.
- Replay buffer: when length reaches 15,000, halve it (remove oldest).
- Training starts once buffer has ≥ 64 transitions.

### 2.9 Key Results

**Zero noise (σ = 10^{-9}):** IS scatter concentrates in the **supra-competitive area** (between Nash and Pareto-optimal IS). Agents trade at different speeds (one fast, one slow), but the **average strategy ≈ TWAP** (Pareto-optimal).

**Moderate noise (σ = 10^{-3}):** Roughly half of runs in supra-competitive area; the rest show **predatory behavior** — the fast trader achieves lower IS at the expense of the slow trader. Still consistent with the Pareto-efficient set.

**Large noise (σ = 10^{-2}):** Similar distribution but with more variance. Centroids still concentrated in supra-competitive area. Average strategy still revolves around TWAP.

**Combined (Figure 7):** Centroids cluster near the Pareto-optimal IS **irrespective of volatility level**. Zero noise: closest to Pareto-optimum. Moderate/Large noise: roughly half in supra-competitive area, rest near Pareto front.

**Average strategies (Figure 8):** Zero noise: slightly **front-loaded** (sell more early). Moderate/Large noise: **less aggressive at beginning**, larger selling rate towards end (uncertainty from volatility).

### 2.10 Variable Volatility (Section 4.5)

**Train zero noise → test large noise:** Strategies resemble those from training/testing both at zero noise. Supra-competitive strategies are still attainable.

**Train large noise → test zero noise:** Strategies resemble those from training/testing both at large noise. Again supra-competitive.

**Key conclusion**: What matters most is the **training phase environment**. Once agents learn supra-competitive behavior in one volatility regime, they still behave supra-competitively in a different regime. The behavior is **robust to misspecified dynamics**.

### 2.11 Is It Tacit Collusion? (Section 5)

The authors use Harrington's definition: "Collusion is when firms use strategies that embody a reward-punishment scheme."

**Their conclusion**: They **cannot conclude** that the observed behavior is tacit collusion in Harrington's sense. There is no clear evidence of a reward-punishment mechanism. However, the supra-competitive behavior could cause **welfare reduction** for other market participants: in the Nash equilibrium, agents trade more at the beginning so information is incorporated faster; when colluding (TWAP), execution is spread more evenly, so **private information is incorporated more slowly**, distorting the price formation process.

### 2.12 Conclusions and Future Work

1. Pareto-optimal (collusive) strategy for risk-neutral agents = TWAP.
2. RL agents find **supra-competitive strategies** (lower cost than Nash, higher than collusion) across all volatility regimes.
3. Agents trade at **different speeds**, adjusting based on the other agent.
4. Supra-competitive behavior is **robust** to misspecified volatility.

**Future work they suggest:**

1. More than two agents and/or more assets (citing Cordoni & Lillo 2024).
2. **Time-varying liquidity** (citing their own Macri & Lillo 2024).
3. Non-linear impact models.
4. **Transient (rather than permanent) price impact** — the Almgren-Chriss model assumes permanent fixed impact, but empirical results point toward transient nature. They note that Nash equilibrium under transient impact shows **price instabilities** (Schied & Zhang 2019), similar to market manipulations. Studying whether different manipulation practices arise with RL would interest regulators.

### 2.13 Replication Notes — What We Need to Replicate

The DDQL implementation is our starting point. Key choices to replicate exactly:

- 5-layer FC network, 30 hidden nodes, LeakyReLU, ADAM lr=0.0001
- Continuous action space (action as Q-network input, not discrete grid)
- Gaussian exploration: μ = q_t/(N-t), δ = |q_t/(N-t)|
- ε-greedy: ε₀ = 1, decay ×0.995 every 75 actions
- Target net hard copy every 75 actions
- Replay buffer max 15,000, halve when full
- γ = 1 (undiscounted)
- Coin toss for trading order at each step

**Key modifications for our work:**
- Asymmetric roles: defender has q_0 = 100, predator has q_0 = 0 (or opposite-signed)
- Predator reward: profit from front-running, not IS minimization
- State may need to include spread/order book features (not just mid-price)
- Action space: limit orders at price levels, not just quantities

---

## Paper 3: Schied & Zhang (2017) — Analytical Nash Equilibrium for Multi-Agent Liquidation

**Reference:** "A state-constrained differential game arising in optimal portfolio liquidation." Mathematical Finance 27(3):779-802, 2017. arXiv:1312.7360.

**Why critical:** This provides the **analytical Nash benchmark** for the Almgren-Chriss multi-agent game. Lillo & Macri cite this as the ground truth against which they compare their RL agents. For our predator-defender game, the asymmetric case (X₁ ≠ X₂) is the relevant one.

### 3.1 Model: n Risk-Averse Agents in Almgren-Chriss (Continuous Time)

**Unaffected price:** S⁰(t) = S₀ + σW(t) + ∫₀ᵗ b(s)ds

**n-agent transaction price:**

```
S^{X₁,...,Xₙ}(t) = S⁰(t) + γ Σⱼ (Xⱼ(t) - Xⱼ(0)) + λ Σⱼ Ẋⱼ(t)
```

where γ ≥ 0 = permanent impact, λ > 0 = temporary impact.

**Each agent i minimizes:** E[C_i] + (α_i/2)·Var[C_i]

where α_i ≥ 0 is risk aversion, and C_i is the implementation shortfall cost:

```
C_i = -∫₀ᵀ Ẋ_i(t) S(t) dt + λ ∫₀ᵀ Ẋ_i(t)² dt
```

**Boundary conditions (state constraints):** X_i(0) = x_i, X_i(T) = 0 for all i.

### 3.2 Theorem 2.2 — Existence and Uniqueness

For any n, T > 0, α_i ≥ 0, and x₁,...,xₙ: there exists a **unique Nash equilibrium** (X₁*,...,Xₙ*) in the class of deterministic absolutely continuous strategies satisfying X_i(T) = 0.

Each X_i* satisfies the coupled ODE system:

```
α_i σ² X_i(t) - 2λ Ẍ_i(t) = b(t) + γ Σⱼ≠ᵢ Ẋⱼ(t) + λ Σⱼ≠ᵢ Ẍⱼ(t)
```

with X_i(0) = x_i, X_i(T) = 0.

**Remark:** This unique mean-variance Nash equilibrium is also a Nash equilibrium for CARA exponential utility within the class of deterministic strategies.

### 3.3 Theorem 2.5 — Closed-Form for Equal Risk Aversion

When α₁ = ... = αₙ = α > 0 and b = 0:

**Individual characteristic exponent:**

```
θ̂ = √(γ² + 4ασ²λ) / (2λ)
```

**Collective characteristic exponent:**

```
ρ̂ = √((n-1)²γ² + 4(n+1)ασ²λ) / (2(n+1)λ)
```

Roots: θ± = γ/(2λ) ± θ̂, ρ± = -(n-1)γ/(2(n+1)λ) ± ρ̂

The i-th equilibrium strategy:

```
X_i*(t) = c_i(θ₊) e^{θ₊t} + c_i(θ₋) e^{θ₋t} + c(ρ₊) e^{ρ₊t} + c(ρ₋) e^{ρ₋t}
```

where c_i(θ±) are agent-specific (depend on x_i - x̄_n) and c(ρ±) are common to all agents (depend on x̄_n).

**Eigenspace structure:**
- ρ± eigenspace = span{(1, ρ±1)} — **symmetric** (collective mode, along 1)
- θ± eigenspace = {v ∈ ℝⁿ : v ⊥ 1} — **antisymmetric** (relative mode)

### 3.4 Corollary 2.6 — Two-Player Explicit Solution

For n = 2 with equal risk aversion:

```
X₁*(t) = ½(Σ(t) + Δ(t))
X₂*(t) = ½(Σ(t) - Δ(t))
```

**Symmetric (collective) part:**

```
Σ(t) = (x₁+x₂) · e^{-γt/(6λ)} · sinh((T-t)√(γ²+12αλσ²)/(6λ)) / sinh(T√(γ²+12αλσ²)/(6λ))
```

**Antisymmetric (relative) part:**

```
Δ(t) = (x₁-x₂) · e^{γt/(2λ)} · sinh((T-t)√(γ²+4αλσ²)/(2λ)) / sinh(T√(γ²+4αλσ²)/(2λ))
```

**For risk-neutral agents (α = 0):** The Nash strategy simplifies further. The permanent impact κ now enters the optimal inventory formula (unlike the single-agent case). For equal inventories (x₁ = x₂), Δ = 0 and both agents use the same strategy.

### 3.5 Qualitative Properties

**Proposition 2.8:** If x₁ ≥ x₂ ≥ 0, then X₁*(t) is strictly decreasing in ασ² for 0 < t < T. More risk aversion → more front-loading.

**Proposition 2.9:** If x₁ = x₂ ≥ 0, then X₁*(t) = X₂*(t) is strictly decreasing in γ and strictly increasing in λ for 0 < t < T. Higher permanent impact → faster selling; higher temporary impact → slower selling.

**When monotonicity breaks down:** For asymmetric positions (x₁ ≠ x₂), monotonicity in γ, λ, and ασ² can all fail. A small agent may *increase* mid-interval inventory when the large agent adjusts — the "competing incentive" effect.

**Financial interpretation:** Permanent impact from one agent's trades is perceived by other agents as an additional price trend. This is why Nash strategies depend on γ while the single-agent strategy does not.

### 3.6 Connection to Our Predator-Defender Game

- **Analytical baseline:** When predator has x₁ = 0 (no inventory) and defender has x₂ = q₀ > 0, the Nash equilibrium becomes: X₁* ≈ 0, X₂* ≈ single-agent Almgren-Chriss solution. The predator has no reason to trade in the Almgren-Chriss model without a distress signal.
- **Why Almgren-Chriss is insufficient for predator-prey:** The predator's profit comes from **front-running** the defender's trades (selling before, buying back after). In the Almgren-Chriss open-loop game, there is no information asymmetry or forced liquidation trigger — so the predator has no informational advantage to exploit. This is exactly why we need **Brunnermeier-Pedersen style asymmetry** (forced liquidation signal) combined with **RL-learned strategies** in a **realistic LOB**.
- **Schied-Zhang 2019 (transient impact):** Under transient price impact, when θ < θ* = G(0)/4, the Nash equilibrium exhibits **hot-potato oscillations** (alternating buy/sell). This is the mathematical analogue of the Flash Crash. Our LOB simulator should naturally produce transient impact, so this instability regime is relevant.

---

## Paper 4: Schied & Zhang (2019) — Market Impact Game under Transient Price Impact

**Reference:** "A market impact game under transient price impact." Mathematics of Operations Research 44(1):102-121, 2019. arXiv:1305.4013.

### 4.1 Model: Two Agents, Transient Impact, Discrete Time

Two risk-neutral agents liquidating positions X₀ and Y₀ over discrete time grid 𝕋 = {t₀,...,t_N}.

**Transient price impact (Bouchaud-Farmer-Lillo kernel):**

```
S_t^Ξ = S_t⁰ - Σ_{t_k < t} G(t - t_k)(ξ₁,k + ξ₂,k)
```

where G: ℝ₊ → ℝ₊ is convex, nonincreasing, nonconstant, strictly positive definite.

**Cost function:** At each t_k, execution priority determined by fair coin flip ε_k ~ Bernoulli(1/2). The cost includes a latency term ε_k · G(0) · ξ₁,k · ξ₂,k and quadratic transaction cost θ · ξ²_{1,k}.

### 4.2 Nash Equilibrium (Theorem 2.5)

For any strictly positive definite G, any 𝕋, θ ≥ 0, and X₀, Y₀ ∈ ℝ: there exists a unique Nash equilibrium:

```
ξ*₁ = ½(X₀ + Y₀)v + ½(X₀ − Y₀)w
ξ*₂ = ½(X₀ + Y₀)v − ½(X₀ − Y₀)w
```

where:

```
v = [(Γ_θ + Γ̃)⁻¹ e] / [eᵀ(Γ_θ + Γ̃)⁻¹ e]     (symmetric/cooperative component)
w = [(Γ_θ − Γ̃)⁻¹ e] / [eᵀ(Γ_θ − Γ̃)⁻¹ e]     (antisymmetric/competitive component)
```

Γ_θ = Γ + 2θI (regularized kernel), Γ̃ = lower-triangular part of Γ plus half-diagonal.

### 4.3 Stability Threshold and Hot-Potato Game (Theorem 2.7)

**Critical threshold:** θ* = G(0) / 4

- If **θ < θ***: The equilibrium strategies exhibit **oscillations** — consecutive buy and sell orders — for all X₀ ≠ Y₀. This is the **hot-potato game** / transaction-triggered price manipulation.
- If **θ ≥ θ***: The equilibrium strategies are **monotone** (no oscillations).

**Mechanism:** When θ is low, the latency cost G(0)ξ₁,kξ₂,k/2 is not sufficiently penalized. Each agent tries to front-run the other by alternating buy/sell to exploit the transient impact decay, passing positions back and forth like a "hot potato."

**Connection to Flash Crash:** Schied & Zhang explicitly interpret the oscillatory regime as the mathematical analogue of the "hot potato" trading observed among HFTs during the May 6, 2010 Flash Crash (CFTC report, Kirilenko et al.).

### 4.4 Counterintuitive Cost Behavior

For certain parameter ranges, **increasing θ can decrease expected costs for both agents**. Higher θ suppresses the oscillatory hot-potato behavior, which is more costly than the direct transaction cost.

### 4.5 High-Frequency Limit (Schied-Strehle-Zhang 2017, arXiv:1509.08281)

- θ = 0, N → ∞: Oscillations persist indefinitely, no continuous-time limit exists.
- θ > 0, N → ∞: Converges to continuous-time NE with θ = θ* = G(0)/4, **regardless** of the discrete-time θ.
- For θ ≠ θ* in continuous time, Nash equilibria typically **do not exist**.

### 4.6 Extension to J Agents (Luo-Schied)

The unique Nash equilibrium for J agents with risk aversion γ:

```
ξ*_{j,·} = X̄ · v + (X_j − X̄) · w
```

where X̄ = (1/J)Σⱼ Xⱼ and:

```
v = [(Γ^{γ,θ} + (J-1)Γ̃)⁻¹e] / [eᵀ[Γ^{γ,θ} + (J-1)Γ̃]⁻¹e]
w = [(Γ^{γ,θ} − Γ̃)⁻¹e] / [eᵀ[Γ^{γ,θ} − Γ̃]⁻¹e]
```

with Γ^{γ,θ}_{i,j} = (Γ_θ)_{i,j} + γ·φ(t_{i-1} ∧ t_{j-1}) (risk-adjusted kernel).

### 4.7 Multi-Asset Extension (Cordoni & Lillo 2020, arXiv:2004.03546)

J agents trade M assets with cross-impact matrix Q ∈ ℝ^{M×M} (symmetric positive definite).

**Key result:** Diagonalize Q = VΛVᵀ → transform to "virtual assets" → solve J independent single-asset games → transform back.

**Per-asset stability threshold:** θ*_m = λ_m · G(0) / 4 (where λ_m is an eigenvalue of Q). Overall stability: θ ≥ max_m θ*_m = λ_max(Q) · G(0)/4.

**Arbitrageur result (Proposition 4.1):** When aggregate net order flow is zero (Σⱼ X_{i,j} = 0 for all i), no arbitrageur can profit — the fundamentalists internalize each other's impact.

### 4.8 Relevance to Our Work

1. **Baselines:** The Schied-Zhang Nash equilibrium is our analytical benchmark for the Almgren-Chriss sub-case.
2. **Instability prediction:** θ* = G(0)/4 threshold predicts when the analytical NE exhibits hot-potato oscillations. In our LOB simulator, the "effective θ" is determined by the bid-ask spread + fees.
3. **Asymmetry recovery:** When predator and defender have different inventories (X₁ ≠ X₂), the antisymmetric component ½(X₀-Y₀)w is nontrivial. This is where BP-style predatory behavior should emerge analytically.
4. **Tacit collusion boundary:** Lillo-Macri find RL agents collude when symmetric. Our asymmetric setting should break this collusion.
5. **Regime dependence:** θ, γ, and G parameters map to our regime axes (spread ↔ θ, volatility ↔ σ, defender size ↔ x₂).

---

## Paper 5: Brunnermeier & Pedersen (2005) — Predatory Trading

**Reference:** "Predatory Trading." Journal of Finance 60(4):1825-1863, 2005.

**Why critical:** This is the **foundational paper** on predatory trading. It defines the predator-prey dynamic in financial markets: a distressed trader is forced to liquidate, and strategic predators exploit this by selling first and buying back later. Our paper directly extends this framework from analytical to RL-learned strategies and from reduced-form to realistic LOB.

### 5.1 Model Setup

**Agents:**

- **I large strategic traders** (arbitrageurs): Risk-neutral, act strategically, have market impact. Each has position limit x̄ (capital constraint).
- **Long-term investors:** Price-takers with downward-sloping aggregate demand: Y(p) = (1/λ)(μ − p). They don't attempt to profit from price swings.

**Distinction:**

- **Prey (I^d):** Distressed traders who *must* liquidate. Driven by margin calls, portfolio insurance, risk management rules, bond downgrades.
- **Predators (I^p):** Unaffected strategic traders who exploit the distressed traders' need to sell.

### 5.2 Forced-Liquidation Mechanism

**Exogenous distress (Section 4.1):** A subset I^d is known to be in distress. A distressed trader must liquidate at minimum speed A/I until position reaches zero.

**Endogenous distress (Section 4.2):** Trader i must liquidate if mark-to-market wealth W^i(t) = x_i(t)·p(t) + O_i(t) drops to threshold W̄. **Key insight:** W(I^d) is **increasing in I^d** — the more traders expected to be in distress, the harder it is for any individual to survive, because more distressed selling → larger price decline → fewer predators (more prey) → fiercer predation → even larger price decline. This creates a **cascade/ripple effect**.

### 5.3 Predator Strategies

**Single predator (Proposition 1):** Two phases:

1. **Sell simultaneously** with the distressed trader at speed A/I
2. **Buy back** at speed A until reaching capacity x̄

The predator sells *even when price is below long-run level* because the price will drop further as long as the distressed trader is still selling. Each marginal share sold high and bought back low is profitable.

**Multiple predators (Proposition 2):** Three phases:

1. **Sell** at speed A/I (shorter than single-predator case)
2. **Buy back** at speed A/[I(I^p−1)] until distressed traders finish
3. **Hold** at x̄

Competition to buy back earlier forces predators to start buying *before* the distressed traders finish selling.

**True front-running (Proposition 8):** If distressed traders start selling at t₁ > t₀ (delayed reaction), predators sell *before* the distressed traders begin! This is the most profitable scenario and the most costly for distressed traders.

### 5.4 Price Model

**P = μ − λ(S − X(t))** where S = aggregate supply, X(t) = total strategic trader holdings, λ = market illiquidity.

This is **not** the Almgren-Chriss model. The permanent impact comes from a *downward-sloping demand curve of long-term investors* (akin to Kyle's λ), and the temporary impact is a *hard capacity constraint* on trading speed, not a smooth quadratic cost. In BP, temporary impact is a *constraint* (infinite penalty beyond A), while in Almgren-Chriss it's a smooth quadratic cost.

### 5.5 Key Theorems

| Proposition | Content |
|-------------|---------|
| Prop 1 | Single predator: sells simultaneously with distressed, then buys back. Effective illiquidity = Iλ (vs. normal λ). |
| Prop 2 | Multiple predators: sell initially, buy back *before* distressed finishes. Price overshoots less. |
| Prop 2' | If x(t₀) < [(I^p−1)/(I−1)]x̄ (lots of sidelined capacity), no predation — predators buy immediately. |
| Prop 3 | Price overshooting: strictly positive for finite I^p; decreasing in I^p; → 0 as I^p → ∞ |
| Prop 4 | Survival hurdle W(I^d) increasing in I^d — more expected distress makes survival harder |
| Prop 6 | Distressed liquidation value increases with number of predators but remains strictly below orderly value even as I^p → ∞ |
| Prop 8 | Front-running equilibrium: predators sell *before* distressed traders start |
| Prop 9 | Batch auction reduces overshooting; predators *buy* (not sell) in the auction |

### 5.6 Liquidity Spiral

1. **Wealth shock** → trader D must liquidate
2. **Predators sell too** → price drops *more* than warranted by distressed selling alone
3. **Lower price** → erodes wealth of other vulnerable traders
4. **More traders breach** threshold → become distressed
5. **Fiercer predation** → price drops further
6. **Repeat** → systemic crisis

This differs from Kyle-Xiong (2001) wealth-effect contagion because **predatory amplification** is the key additional mechanism.

### 5.7 Policy Implications

- **Batch auctions / circuit breakers** (Proposition 9): Prevent predators from walking down the demand curve. Predators *buy* in the auction, not sell.
- **Up-tick rule**: Prevents "bear raids" — predators with small positions cannot short-sell during a declining market.
- **Disclosure**: Strict disclosure of positions *increases* predation risk (predators know exactly what to sell and how long to prey). Should be dispersed broadly (more predators → less fierce predation) and delayed.
- **Central bank intervention**: Bailout of 1-2 traders can stabilize prices and save numerous other vulnerable traders.

### 5.8 How BP Differs from Almgren-Chriss Multi-Agent

| Feature | BP (2005) | Almgren-Chriss Multi-Agent |
|---------|-----------|---------------------------|
| Objective | Maximize profit by exploiting *others'* price impact | Minimize own execution cost |
| Impact model | Permanent: demand curve P = μ − λ(S−X); Temporary: hard speed constraint A | Permanent: linear (κ); Temporary: quadratic (α·v²) — smooth |
| Traders' motivation | Profit from foreknowledge of future order flow | Optimal execution of own exogenous orders |
| Asymmetric roles | Distressed (must sell) vs. predators (choose to sell then buy) | All agents are symmetric executioners |
| Equilibrium type | Nash with heterogeneous constraints | Nash among symmetric cost-minimizers |
| Price overshooting | Central result | No overshooting |
| Liquidity provision | Endogenous: predators *withdraw* liquidity when needed most | Not modeled |

**Bottom line:** BP is about *strategic exploitation* of another's distress; Almgren-Chriss multi-agent is about *competitive equilibrium* among execution-cost-minimizing agents. The predator's profit comes from the *distressed trader's* impact, not their own.

### 5.9 Relevance to Our Work

- **Predator motive:** BP provides the economic justification for why a predator exists — they profit from foreknowledge of the defender's forced liquidation. Our predator agent should be trained to maximize this profit, not minimize its own execution cost.
- **Front-running and buy-back:** The predator's optimal strategy has two phases: sell before/during the defender's liquidation, then buy back at depressed prices. Our RL agent should discover this naturally.
- **Liquidity spiral:** The cascade mechanism is relevant to our regime analysis — in high-stress regimes (low liquidity, large defender position), predation should be more severe.
- **Policy angle:** Our results on the effectiveness of defender strategies under predation could inform real-world policy (circuit breakers, dark pools, etc.).

---

## Paper 6: Carlin, Lobo, & Viswanathan (2007) — Episodic Liquidity Crises

**Reference:** "Episodic Liquidity Crises: Cooperative and Predatory Trading." Journal of Finance 62(5):1997-2041, 2007.

**Why critical:** CLV extends BP 2005 by adding transaction costs (the λY_t term), which fundamentally changes the predator's strategy from parallel trading to **racing-and-fading**. The repeated-game framework and episodic illiquidity result are directly relevant to our regime-conditional analysis.

### 6.1 Model: Price with Transaction Cost Term

```
P_t = U_t + γX_t + λY_t
```

where U_t = fundamental (martingale), γX_t = inventory effect (strategic traders' aggregate holdings), **λY_t = transaction cost / flow-impact term** (aggregate rate of trading).

The λY_t term is **absent in BP (2005)**. This is the key addition.

### 6.2 How This Transforms the Predator Strategy

In BP (only inventory matters): Predator trades at the *same rate* as the distressed trader (timing indeterminate).

In CLV (trading rate matters): Predator **races** (front-loads selling, because selling faster pushes price down more via λY_t) then **fades** (buys back at depressed prices).

### 6.3 Key Results in the Stage Game

**Result 1 (Monopolist):** Sells at constant rate Y_t = Δx/T. Value: V₁ = -uΔx - (γ/2 + λ/T)Δx².

**Result 5 (Multiple distressed, no predators):** Symmetric equilibrium — traders race in *decreasing exponential* fashion: Y^i_t = ae^{-(n-1)/(n+1) · γ/λ · t}. More traders → earlier selling.

**Result 6 (1 distressed + 1 predator):** Unique equilibrium:

```
Y^d_t = ae^{-⅓γ/λ·t} + be^{γ/λ·t}     (distressed)
Y^p_t = ae^{-⅓γ/λ·t} - be^{γ/λ·t}     (predator)
```

The predator **races** (Y^p > 0 early) then **fades** (Y^p < 0 later, buying back).

### 6.4 Surplus and Efficiency

**Total surplus loss from competition/predation:** ΔV_n = V₁ - V_n is *monotonic increasing in T, γ, n* and *monotonic decreasing in λ*.

- As λ→0: ΔV_n → ½(n-1)/(n+1)·γΔx² — purely from inventory competition.
- As λ→∞: ΔV_n → 0 — huge transaction cost deters all aggressive trading.

**Two-trader case (Result 7):** As λ→0, predator gains ⅙γΔx² while distressed seller *loses* ⅓γΔx² — **the predator gains half of what the distressed trader loses** (the other half is deadweight loss). Predation creates a pure **deadweight loss**, not just a transfer.

### 6.5 Repeated Game — Cooperation vs. Predation (Result 9)

**Trigger strategy** (grim trigger): Cooperation in each period, with deviation punished by reverting to the non-cooperative stage game forever.

**Critical discount factor:**

```
δ ≥ V_p / [p₁₀(V₁ - V_d) + ½p₁₁(V₁(2Δx) - V₂(2Δx)) + (1-p₀₁)V_p]
```

If p₀₁ ≥ 2(p₁₀ - p₁₁), **no δ supports cooperation** — the incentive to predate when the other is distressed is too strong.

### 6.6 Episodic Illiquidity (Result 11)

When Δx is stochastic, there exists a **threshold Δx̄** such that:

- |Δx| < Δx̄: traders **cooperate**
- |Δx| > Δx̄: **predation occurs**, but cooperation *resumes* next period

This yields **episodic illiquidity**: markets are stable and liquid most of the time, but break down when shocks are large enough. The threshold depends on C/K (ratio of future cooperation value to predation profit). For normal shocks: minimum C/K ≈ 4.67; threshold Δx̄ ≈ 1.37σ.

### 6.7 Contagion Across Markets (Result 12)

**Multimarket contact** makes cooperation *easier* to sustain (more at stake, deviation punished across all markets). Required δ for cooperation decreases with number of markets.

However, once a liquidity event is large enough to trigger predation, **all markets become illiquid simultaneously** — this is contagion.

| n markets | Min C/K for cooperation |
|-----------|------------------------|
| 1 | 4.67 |
| 2 | 3.35 |
| 5 | 2.40 |
| 8 | 2.09 |
| 20 | 1.70 |

### 6.8 How CLV Extends BP

| Feature | BP (2005) | CLV (2007) |
|---------|-----------|------------|
| Price equation | P = U + γX (inventory only) | P = U + γX + λY (inventory + flow impact) |
| Predator strategy | Trade at same rate as distressed | Race then fade (time-varying) |
| Price dynamics | No jumps during predation | Price jumps during predation |
| Transaction costs | Not modeled | Endogenous (quadratic, via λY) |
| Trading volume | Constant | U-shaped (consistent with data) |
| Repeated interaction | One-shot game | Infinitely repeated with trigger strategies |
| Cooperation | Not analyzed | Central: cooperative equilibrium is Pareto superior |
| Episodic illiquidity | Not explained | Endogenously generated |
| Contagion | Not modeled | Multimarket contact → easier cooperation but contagion when it breaks |

### 6.9 Relevance to Our Work

- **Racing-and-fading:** Our RL predator should discover this strategy naturally in the LOB simulator. The LOB's discrete price levels and order queue dynamics provide the "transaction cost" analog (λY) that makes racing-and-fading optimal.
- **Episodic illiquidity:** Our regime-conditional analysis is the empirical analogue of CLV's threshold result. In high-stress regimes (large defender position, wide spread), predation should be triggered; in low-stress regimes, cooperation might emerge.
- **Deadweight loss:** CLV shows that predation is not just a transfer — it creates real welfare loss. Our work can quantify this in the LOB setting by comparing total market surplus with and without predators.
- **Contagion:** If we extend to multi-asset settings, CLV's multimarket results predict that predation episodes should be correlated across assets.

---

## Paper 7: Mascioli et al. (2024) — PyMarketSim

**Reference:** "A Financial Market Simulation Environment for Trading Agents Using Deep Reinforcement Learning." ICAIF 2024 (**Best Paper Award**). Authors: Chris Mascioli, Anri Gu, Yongzhao Wang, Mithun Chakraborty, Michael P. Wellman.

**Why critical:** PyMarketSim is our **simulator**. This paper describes its architecture, capabilities, and the EGTA methodology used with it.

### 7.1 Architecture

Five core components:

```
Simulator (top-level orchestrator)
├── EventQueue — discrete event-driven scheduling
├── agents: dict[int, Agent] — heterogeneous trading agents
└── markets: list[Market]
    ├── Fundamental — latent asset value process (mean-reverting)
    ├── FourHeap — 4-heap limit order book
    └── matched_orders — transaction log
```

**Single timestep flow:**

1. EventQueue fires — agents scheduled at time t arrive
2. Each agent's `take_action()` returns `list[Order]`
3. Market inserts orders into 4-Heap LOB
4. `market.clear_market()` matches compatible buy/sell, records transactions
5. Simulator calls `agent.update_position(q, cash)` on matched agents
6. Advance to next scheduled event

### 7.2 Four-Heap Order Book

Based on **Wurman 1998**. Four heaps:

| Heap | Structure | Contents |
|------|-----------|----------|
| B_in (buy_matched) | min-heap | Matched buy orders (price ≥ best sell) |
| B_out (buy_unmatched) | max-heap | Unmatched buy orders waiting for seller |
| S_in (sell_matched) | max-heap | Matched sell orders (price ≤ best buy) |
| S_out (sell_unmatched) | min-heap | Unmatched sell orders waiting for buyer |

**Complexity:** Insert O(log n), Quote O(1), Clear O(|matched|). Benchmarked: **<0.005ms per operation at book size 10^6**.

**Insert logic:** For a SELL order: if price ≤ buy_unmatched.peek() and sell_matched.peek() ≤ buy_unmatched.peek() → **new match**; elif price ≤ sell_matched.peek() → **replace** existing matched order; else → add to sell_unmatched.

### 7.3 Agent Models

**ZIAgent (Zero Intelligence):** Random side (BUY/SELL), estimates final fundamental f̂_T = (1-ρ)·mean + ρ·f_t where ρ = (1-r)^(T-t), draws random surplus demand from [shade[0], shade[1]], price = estimate + pv_value ∓ valuation_offset.

**Informed ZI Agent:** Same as ZI but uses market.get_final_fundamental() — the TRUE final value. Information advantage.

**HBLAgent (Heuristic Belief Learning):** Maintains memory of last L matched orders. Computes belief function: probability an order at price p will transact. Uses spline interpolation + scipy optimization to find price maximizing expected surplus. Requires `fastcubicspline`.

**MMAgent (Market Maker — Ladder Policy):** Places K levels on each side: buy at bt - k·ξ, sell at st + k·ξ. Where bt = min(estimate - ω/2, best_ask), st = max(estimate + ω/2, best_bid). Quantity = 1 per level.

**MMAgent Beta (Beta Policy):** Generalizes MM strategies using scaled Beta distribution. Three modes: (1) static beta params, (2) RL policy (action = 4-tuple of beta params), (3) **inventory-driven policy** that shifts volume to reduce inventory.

**SpoofingAgent:** Takes action = (regular_order_price, spoofing_order_price). Places genuine SELL + large-volume deceptive BUY (spoof).

### 7.4 Arrival Model

Two modes:

1. **Bernoulli arrival** (base Simulator): each agent arrives with probability λ per step.
2. **Sampled arrival** (SimulatorSampledArrival_MM): pre-samples geometric inter-arrival times via `torch.distributions.Geometric`, scheduling agents into a `defaultdict(list)` keyed by timestep. MM gets separate higher arrival rate λ_MM.

### 7.5 How PyMarketSim Differs from ABIDES

| Simulator | Strength | Weakness |
|-----------|----------|----------|
| ABIDES (JPMorgan) | Very large agent counts | Slow for RL workflows |
| JAX-LOB | GPU-accelerated | No agent infrastructure |
| FinRL | Many real datasets | No microstructure model |
| **PyMarketSim** | **Fast CPU-parallel, RL-native, realistic microstructure** | Moderate agent counts |

**Key differentiators:** Event-driven architecture (skipping empty timesteps), 4-Heap O(log n) LOB, CPU-parallelism inspired by Sample Factory/IMPALA. Processes **~300,000 LOB operations/second** and **~15,000 agent actions/second**. Has **Gym-style RL wrappers** (`marketsim.wrappers`) — ABIDES lacks native RL integration.

### 7.6 EGTA Application

The paper studies a **multi-agent market game** with **TRON agents** (Trained Response Order Networks):

- Recurrent neural networks (LSTM + dueling DQN, following **R2D2 architecture**)
- Condition on: current BID/ASK, time remaining, holdings, private values
- Outperform ZI agents by **8-12% more surplus** across market environments

**EGTA method: PSRO:**

1. Start with a population of strategies (e.g., ZI agents with different shade parameters)
2. Train a TRON agent as **best response** to the current population
3. Add the best response to the population
4. Compute the **empirical payoff matrix** by simulating all strategy combinations
5. Analyze for **approximate Nash equilibria**
6. Iterate

### 7.7 Performance Metrics (from 13 analytical plots)

Price discovery (mid-price vs. fundamental), bid-ask spread, rolling volatility, autocorrelation of returns, transaction volume, price impact, PnL distribution, inventory over time, surplus extraction, PnL comparison vs. ZI baseline, lookback sweep, shock variance sweep, convergence.

**Conservation laws verified:** Position conservation (Σ positions = 0) and Cash conservation (Σ cash = 0).

### 7.8 Relevance to Our Work

- **Our simulator:** We will use PyMarketSim directly. The 4-heap LOB, event-driven architecture, and RL wrappers are exactly what we need.
- **Agent design:** Our defender agent will be similar to a TRON agent (RL-trained execution policy). Our predator agent will be a new agent type designed to exploit forced liquidation.
- **EGTA pipeline:** We will follow the PSRO methodology from this paper, but applied to the predator-defender game instead of the market-making game.
- **Background agents:** ZI agents and MM agents will serve as background market participants, providing liquidity and noise trading.
- **Limitation:** PyMarketSim's moderate agent count is fine for our purposes (1 defender + 1-3 predators + ~100 background ZI + 1-3 MMs).

---

## Cross-Paper Synthesis

### How the Papers Relate to Each Other

```
BP 2005 ──── predator front-runs and buys back
  │              (demand curve model, no transaction cost)
  │
  ▼
CLV 2007 ──── predator races and fades
  │              (adds λY transaction cost, repeated game)
  │
  ▼
Schied-Zhang 2017 ──── analytical Nash for n-agent Almgren-Chriss
  │              (continuous-time, mean-variance, closed-form for equal α)
  │
  ▼
Schied-Zhang 2019 ──── analytical Nash for transient impact
  │              (hot-potato instability when θ < G(0)/4)
  │
  ▼
Lillo-Macri 2024 ──── RL agents learn supra-competitive strategies
  │              (DDQL, symmetric game, tacit collusion)
  │
  ▼
Our paper ──── asymmetric predator-defender in realistic LOB
               (EGTA + PyMarketSim + regime-conditional analysis)
```

### What Changes When We Introduce Asymmetry

| Feature | Lillo-Macri (symmetric) | Our paper (asymmetric) |
|---------|------------------------|----------------------|
| Both agents liquidate | Yes — both sell | No — only defender sells; predator front-runs then buys back |
| Nash benchmark | Schied-Zhang with x₁ = x₂ | Schied-Zhang with x₁ = 0, x₂ = q₀ (predator has no inventory) |
| Pareto-optimal | TWAP (both sell evenly) | Not TWAP — predator should sell early, buy late |
| Tacit collusion | Emerges naturally | Should NOT emerge — incentives are opposed |
| Learned equilibrium | Supra-competitive (near TWAP) | Should be near analytical Nash or BP-style predatory |
| State info | g_t = (t, q_t, S_{t-1}) | Need LOB features (spread, depth at BBO) |
| Action space | Continuous quantity | Limit orders at price levels |
| Impact model | Almgren-Chriss (reduced-form) | LOB microstructure (4-heap, discrete price levels) |

### Why the LOB Matters (vs. Almgren-Chriss)

In Almgren-Chriss, the predator's strategy is simple: sell quantity v, pay temporary impact αv². The LOB introduces:

1. **Discrete price levels** — the predator cannot sell at any price; must choose a limit price or cross the spread.
2. **Order queue dynamics** — the predator's order may be queued behind other orders, introducing execution uncertainty.
3. **Bid-ask spread** — the effective "temporary impact" is at least half the spread, which can be regime-dependent.
4. **Market makers** — provide liquidity but can withdraw it (adverse selection), changing the effective impact function.
5. **Price discovery** — the mid-price moves endogenously based on order flow, not just exogenous permanent impact.

These features make the predator's problem richer and the defender's challenge harder. The predator must now reason about *where* to place orders (not just *how much* to sell), and the defender must reason about the LOB's state to minimize predation impact.

### Regime-Conditional Equilibrium Predictions

Drawing on all six papers, our regime axes map to model parameters:

| Regime axis | Model parameter | Prediction |
|------------|-----------------|------------|
| Volatility (σ) | Schied-Zhang: ασ² | Higher σ → more front-loading (risk-averse defender) |
| Spread (s) | Schied-Zhang 2019: θ | Narrow spread (θ < θ*) → potential hot-potato instability; wide spread (θ ≥ θ*) → stable monotone strategies |
| Defender size (q₀) | BP/CLV: Δx | Large q₀ → stronger predation incentive (CLV threshold Δx̄); small q₀ → cooperation possible |
| Number of predators (K) | BP/CLV: I^p | More predators → less overshooting (BP Prop 3) but more total surplus loss (CLV Result 8) |
| Market maker presence | PyMarketSim: MM parameters | MM provides liquidity buffer; withdrawal during stress amplifies predation |

### The Key Hypothesis

**H₁:** In the symmetric Almgren-Chriss game, RL agents converge to supra-competitive (near-TWAP) equilibria (Lillo-Macri finding). **Introducing asymmetric predator-defender roles breaks the tacit collusion** and pushes the learned equilibrium toward the analytical Nash equilibrium (Schied-Zhang) or the BP-style predatory equilibrium, depending on the regime.

**H₂:** The transition from cooperation to predation is governed by regime parameters, with a threshold structure analogous to CLV's Δx̄ threshold. Below the threshold (small defender position, wide spread, high volatility), the predator has insufficient profit opportunity and the equilibrium resembles the cooperative outcome. Above the threshold, predation emerges.

**H₃:** In the realistic LOB setting, the predator's learned strategy exhibits **racing-and-fading** (CLV) rather than the parallel trading of BP, because the LOB's discrete price levels and order queues provide the transaction-cost analog (λY) that makes front-loading optimal.
