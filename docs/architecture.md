# Architecture & Design Notes

## Data flow

```
Simulation.run(steps)
│
├── per step:
│   ├── StochasticProcess.step()        → new fundamental price
│   ├── for each agent: agent.fundamental = new_fund
│   ├── random.choice(agents).act(market)
│   │       └── predict_return(market)  → clipped return signal
│   │       └── market.submit_order(Order)
│   │               └── LimitOrderBook.add_order()
│   │
│   ├── Market.step()
│   │       └── LimitOrderBook.match()  → [(price, vol, bid_id, ask_id)]
│   │       └── update self.price, history, return_history
│   │
│   ├── Simulation._settle(trades)
│   │       └── PnLLedger.record_buy/sell() per matched agent
│   │
│   └── Simulation._snapshot()
│           └── PnLLedger.snapshot(price) → append to net_worth_history
│
└── return SimResult(self)
```

## Key design decisions

### Short-selling is enabled

The original FCN model (Chiarella & Iori 2002) makes no distinction between
opening and closing a position. Agents simply submit orders based on their
return prediction. This means the LOB is two-sided from tick 1.

We enforce soft position limits:
- FCN/Trend/MeanRev/FCL: MAX_LONG=20, MAX_SHORT=−10
- RL agents: RL_MAX_LONG=15, RL_MAX_SHORT=−5 (tighter to focus learning)

### Return clipping prevents price explosions

Without clipping, high-chartist-weight agents (wc drawn from Exp(2) can be ~8)
produce predicted prices that drift exponentially from the current price. When
such orders cross, the LOB mid-price formula averages two extreme values and
the resulting trade price inherits the instability. The fix:

```python
max_r = log(1.10) / horizon   # predicted price stays within ±10%
r = clip(r, -max_r, +max_r)
```

### RL agents use FCN order pricing

Rather than posting at the current mid-price, RL agents still use
`predict_return()` to compute a limit price with a small random margin.
This means RL actions (buy/sell/hold) stay within the same order-price
distribution as rule-based agents, keeping the LOB liquidity consistent.

### Double DQN with feasibility masking

Infeasible actions receive Q = −1e9 before argmax:
- buy when position ≥ RL_MAX_LONG
- sell when position ≤ RL_MAX_SHORT

This ensures the greedy policy never wastes steps on impossible actions,
which would generate zero-reward transitions that pollute the replay buffer.

### TrainingLoop id stability

RL agents are created once before training and their `agent_id` (900, 901)
is the stable dict key in `TrainingLoop.ep_final_worth`. The `build_agents()`
function appends pre-built agents WITHOUT reassigning their ids, preventing
the KeyError that would occur if ids were clobbered to sequential values.

## Stochastic process parameter guide

| Process | Key params | Effect of increasing |
|---|---|---|
| GBM | σ | More volatile paths, wider price range |
| OU | κ | Faster mean reversion |
| OU | σ | Wider oscillations around θ |
| Merton | λ | More frequent jumps |
| Merton | σ_j | Larger jump sizes |
| Heston | ξ | More volatile volatility (vol-of-vol) |
| Heston | ρ | More negative → stronger leverage effect |

## Known limitations

1. **Single asset** — no cross-asset correlation or portfolio effects.
2. **Unit volume orders** — all orders are for exactly 1 share; extending to
   variable volume requires updating the PnLLedger FIFO matching logic.
3. **No transaction costs** — adding a fixed or proportional cost would
   discourage excessive trading and is a straightforward extension.
4. **Euler-Maruyama discretisation** — the Milstein scheme would halve the
   weak error for Heston at coarse time steps.
5. **Tabular Q-table size** — 625 states × 3 actions is small; discretising
   more features or using finer bins quickly becomes intractable without
   function approximation.
