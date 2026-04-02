# 🏦 Agent-Based Financial Market Simulation

**Complex Systems · Stochastic Processes · Reinforcement Learning**

A research-grade, fully self-contained simulation of a double-auction financial market with heterogeneous agents, plug-in stochastic fundamental processes, and reinforcement-learning agents that learn wealth-maximising strategies — all in pure Python / NumPy, no PyTorch required.

Based on the FCLAgent architecture from *Hashimoto et al. (2025): Agent-Based Simulation of a Financial Market with Large Language Models*, extended with RL training and stochastic volatility models.

---

## Key Features

| Layer | What's implemented |
|---|---|
| **Market microstructure** | Price-time priority limit order book · deque price levels · O(1) FIFO matching · bid-ask spread · order flow imbalance |
| **Stochastic fundamentals** | GBM · Ornstein-Uhlenbeck · Merton Jump Diffusion · Heston stochastic volatility |
| **Rule-based agents** | FCNAgent (Fundamental + Chartist + Noise) · TrendFollower (EMA crossover) · MeanReversion (z-score) · FCLAgent (LLM via Ollama) |
| **RL agents** | Tabular Q-Learning (625-state discretisation) · Double DQN (pure NumPy MLP, experience replay, target network) |
| **RL training** | Multi-episode loop · randomised process per episode · portfolio reset · feasibility masking · ε-greedy decay |
| **Analytics** | Stylized facts · ATH anomaly regression (Li & Yu 2012) · Sharpe · max drawdown · 2-state HMM regime detection |

---

## Quickstart

```bash
git clone https://github.com/yberediafe/agent-based-llm-market.git
cd agent-based-llm-market
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/market_sim_full.ipynb
```

### Minimal programmatic usage

```python
from src.abm_market_sim import SimConfig, GBMProcess, Simulation

cfg    = SimConfig(seed=42, n_agents=80, initial_cash=10_000.0)
proc   = GBMProcess(initial_price=300.0, sigma=1e-3)
sim    = Simulation(cfg, process=proc, n_trend=5, n_mean_rev=5)
result = sim.run(steps=500)

print(result.perf_table().head(10))
print(f"Final price: {result.final_price:.2f}")
```

### Train RL agents

```python
from src.abm_market_sim import (
    SimConfig, MertonJumpDiffusion, Simulation,
    QLearningAgent, DQNAgent, TrainingLoop
)

cfg     = SimConfig(seed=42, n_agents=60, n_train_episodes=80, steps_per_episode=350)
ql      = QLearningAgent(agent_id=900, cfg=cfg)
dqn     = DQNAgent(agent_id=901, cfg=cfg, hidden=[64, 64])

trainer = TrainingLoop(
    base_cfg=cfg,
    rl_agents=[ql, dqn],
    process_factory=lambda seed: MertonJumpDiffusion(300.0, lam=0.02),
)
trainer.train()
trainer.plot_training()
```

---

## Project structure

```
agent-based-llm-market/
├── notebooks/
│   └── market_sim_full.ipynb   # Full research notebook (18 sections)
├── src/
│   └── abm_market_sim.py       # Complete Python module (importable)
├── tests/
│   └── test_core.py            # Pytest suite — unit + integration tests
├── docs/
│   └── architecture.md         # Design notes and theoretical background
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Architecture

### Market microstructure

```
Agent.act(market)
    └─ predict_return()          # FCN / strategy signal
    └─ submit_order(Order)
           └─ LimitOrderBook.add_order()

Market.step()
    └─ LimitOrderBook.match()    # price-time priority
    └─ returns [(price, vol, buyer_id, seller_id)]

Simulation._settle(trades)
    └─ PnLLedger.record_buy/sell()   # FIFO cost basis

Simulation._snapshot()
    └─ PnLLedger.snapshot(price)     # net-worth timeline
```

### RL state space (8 features)

| # | Feature | Range |
|---|---|---|
| 0 | ATH nearness $p_t / p^h_{1:t}$ | [0, 2] |
| 1 | ATL nearness $p_t / p^l_{1:t}$ | [0, 2] |
| 2 | Short momentum (10-step) | [−3, 3] |
| 3 | Long momentum (50-step) | [−3, 3] |
| 4 | Rolling volatility | [0, 5] |
| 5 | Normalised position | [−2, 2] |
| 6 | Cash ratio | [−1, 3] |
| 7 | Unrealised PnL ratio | [−1, 1] |

### Action space

| Action | Meaning |
|---|---|
| 0 | Hold — do nothing |
| 1 | Buy 1 share (limit order at FCN-predicted price) |
| 2 | Sell 1 share (short-selling enabled, capped at −10) |

---

## Running the tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=term-missing
```

Expected output: **19 tests pass**, covering the LOB, PnLLedger, stochastic processes, state encoder, and end-to-end simulation integrity.

---

## Enabling LLM agents (optional)

Install [Ollama](https://ollama.com) and pull a model:

```bash
ollama pull llama3
```

Then set `n_llm > 0` in `SimConfig`:

```python
cfg = SimConfig(n_llm=5, llm_model='llama3:latest', llm_cache_enabled=True)
```

Agents fall back to a buy signal gracefully if Ollama is unavailable.

---

## Theoretical background

### FCN Agent (Chiarella & Iori 2002)

Each agent's excess demand is driven by three components:

$$r_i = \frac{1}{w_f + w_c + w_n}\left[\frac{w_f}{200}\ln\frac{p^*}{p_t} + \frac{w_c}{\tau_i}\ln\frac{p_t}{p_{t-\tau_i}} + w_n \varepsilon_t \right]$$

- $w_f \sim \text{Exp}(\lambda_f)$ — fundamental weight  
- $w_c \sim \text{Exp}(\lambda_c)$ — chartist (momentum) weight  
- $w_n \sim \text{Exp}(\lambda_n)$ — noise weight  
- Returns are clipped to $\pm\ln(1.10)/\tau_i$ to prevent LOB momentum cascades

### Heston Model

$$dS = \mu S\,dt + \sqrt{V}\,S\,dW_S$$
$$dV = \kappa(\theta - V)\,dt + \xi\sqrt{V}\,dW_V, \quad \rho = \text{corr}(dW_S, dW_V)$$

Feller condition $2\kappa\theta > \xi^2$ is maintained by parameter defaults.

### Double DQN

The online network selects the action; the target network evaluates the value, reducing overestimation bias:

$$y_i = r_i + \gamma \cdot Q_{\theta^-}\!\left(s_{i+1},\, \arg\max_a Q_\theta(s_{i+1}, a)\right)$$

Implemented in pure NumPy — no PyTorch or TensorFlow required.

---

## Citation

If you use this simulation framework in your research, please cite:

```bibtex
@software{abm_market_sim_2025,
  title   = {Agent-Based Financial Market Simulation with RL Agents},
  year    = {2025},
  url     = {https://github.com/yberediafe/agent-based-llm-market},
  note    = {Based on Hashimoto et al. (2025): Agent-Based Simulation of a
             Financial Market with Large Language Models}
}
```

---

## References

- Chiarella, C. & Iori, G. (2002). A simulation analysis of the microstructure of double auction markets. *Quantitative Finance*, 2(5), 346–353.
- Hashimoto et al. (2025). Agent-Based Simulation of a Financial Market with Large Language Models. *arXiv preprint*.
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility. *Review of Financial Studies*, 6(2), 327–343.
- Li, J. & Yu, J. (2012). Investor attention, psychological anchors, and stock return predictability. *Journal of Financial Economics*, 104(2), 401–419.
- Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, 3(1–2), 125–144.
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529–533.
- van Hasselt, H., Guez, A. & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *AAAI*, 30(1).

---

## License

MIT — see [LICENSE](LICENSE).
