"""
abm_market_sim.py
==================
Agent-Based Financial Market Simulation

Full implementation — classes and helpers extracted from the research notebook.
Import this module to use the simulation programmatically.

Quick start
-----------
    from src.abm_market_sim import SimConfig, GBMProcess, Simulation
    cfg = SimConfig(seed=42, n_agents=80)
    sim = Simulation(cfg, process=GBMProcess(300.0))
    result = sim.run(steps=500)
    print(result.perf_table())
"""

# ---------------------------------------------------------------------------
# Standard-library & third-party imports
# ---------------------------------------------------------------------------
from __future__ import annotations

import hashlib
import json
import random
import uuid
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore
        def __init__(self, it, **kw): self._it = it
        def __iter__(self): return iter(self._it)
        def set_postfix(self, **kw): pass

try:
    from sortedcontainers import SortedDict
except ImportError:
    class SortedDict(dict):  # type: ignore
        def __init__(self, key=None):
            super().__init__()
            self._key = key or (lambda x: x)
        def peekitem(self, idx=0):
            keys = sorted(self.keys(), key=self._key)
            k = keys[idx]; return k, self[k]
        def items(self):
            return [(k, self[k]) for k in sorted(self.keys(), key=self._key)]

warnings.filterwarnings("ignore")


@dataclass
class SimConfig:
    """Single source of truth for all hyperparameters."""
    # Reproducibility
    seed: int = 42

    # Market
    initial_price: float = 300.0
    fundamental_price: float = 300.0

    # Agents
    n_agents: int = 80
    n_llm: int = 0          # set to >0 if Ollama is running
    initial_cash: float = 10_000.0

    # FCN parameters
    lambda_f: float = 10.0
    lambda_c: float = 2.0
    lambda_n: float = 1.0
    horizon_min: int = 10
    horizon_max: int = 100

    # LLM (optional)
    llm_model: str = 'llama3:latest'
    llm_cache_enabled: bool = True

    # RL hyperparameters
    rl_alpha: float = 0.05         # Q-learning rate
    rl_gamma: float = 0.99         # discount factor
    rl_epsilon_start: float = 1.0
    rl_epsilon_decay: float = 0.997
    rl_epsilon_min: float = 0.05
    dqn_lr: float = 5e-4
    dqn_batch: int = 64
    dqn_target_update: int = 100
    replay_capacity: int = 20_000

    # Training loop
    n_train_episodes: int = 60
    steps_per_episode: int = 400

    def seed_everything(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)


DEFAULT_CFG = SimConfig()


class Order:
    """Memory-efficient order with __slots__."""
    __slots__ = ('order_id', 'agent_id', 'side', 'price', 'volume', 'timestamp')

    def __init__(self, order_id, agent_id, side, price, volume, timestamp):
        self.order_id = order_id
        self.agent_id = agent_id
        self.side = side
        self.price = price
        self.volume = volume
        self.timestamp = timestamp

    def __repr__(self):
        return f'Order({self.side.upper()} {self.volume}@{self.price:.2f} agent={self.agent_id})'


class LimitOrderBook:
    """Price-time priority LOB. deque price levels give O(1) FIFO pop."""

    def __init__(self):
        self.bids: SortedDict = SortedDict(lambda x: -x)   # best bid first
        self.asks: SortedDict = SortedDict()                # best ask first
        self.order_map: Dict[str, Order] = {}
        self.trade_log: List[Tuple] = []

    def _book(self, side): return self.bids if side == 'buy' else self.asks

    def add_order(self, order: Order):
        book = self._book(order.side)
        if order.price not in book:
            book[order.price] = deque()
        book[order.price].append(order)
        self.order_map[order.order_id] = order

    def cancel_order(self, order_id: str) -> bool:
        if order_id not in self.order_map:
            return False
        order = self.order_map.pop(order_id)
        book = self._book(order.side)
        lvl = book.get(order.price)
        if lvl:
            try: lvl.remove(order)
            except ValueError: pass
            if not lvl: del book[order.price]
        return True

    def match(self, timestamp: int) -> List[Tuple]:
        """Run matching engine. Returns [(price, vol, buyer_id, seller_id)]."""
        trades = []
        while self.bids and self.asks:
            bp = self.bids.peekitem(0)[0]
            ap = self.asks.peekitem(0)[0]
            if bp < ap: break
            bid_q, ask_q = self.bids[bp], self.asks[ap]
            bid_o, ask_o = bid_q[0], ask_q[0]
            vol = min(bid_o.volume, ask_o.volume)
            tprice = (bp + ap) / 2.0
            trades.append((tprice, vol, bid_o.agent_id, ask_o.agent_id))
            bid_o.volume -= vol; ask_o.volume -= vol
            if bid_o.volume == 0:
                bid_q.popleft()
                self.order_map.pop(bid_o.order_id, None)
            if ask_o.volume == 0:
                ask_q.popleft()
                self.order_map.pop(ask_o.order_id, None)
            if not bid_q: del self.bids[bp]
            if not ask_q: del self.asks[ap]
        self.trade_log.extend(trades)
        return trades

    def spread(self) -> Optional[float]:
        if not self.bids or not self.asks: return None
        return self.asks.peekitem(0)[0] - self.bids.peekitem(0)[0]

    def ofi(self) -> float:
        """Order Flow Imbalance: (buy_vol - sell_vol) / total_vol."""
        bv = sum(sum(o.volume for o in q) for q in self.bids.values()) if self.bids else 0
        sv = sum(sum(o.volume for o in q) for q in self.asks.values()) if self.asks else 0
        tot = bv + sv
        return (bv - sv) / tot if tot > 0 else 0.0

    def depth(self, n=5):
        bids = [(p, sum(o.volume for o in q)) for p, q in list(self.bids.items())[:n]]
        asks = [(p, sum(o.volume for o in q)) for p, q in list(self.asks.items())[:n]]
        return {'bids': bids, 'asks': asks}


class Market:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.lob = LimitOrderBook()
        self.price: float = cfg.initial_price
        self.history: List[float] = [cfg.initial_price]
        self.timestamp: int = 0
        self.spread_history: List[Optional[float]] = []
        self.return_history: List[float] = []

    def submit_order(self, order: Order):
        self.lob.add_order(order)

    def step(self) -> List[Tuple]:
        trades = self.lob.match(self.timestamp)
        if trades:
            new_price = trades[-1][0]
            ret = np.log(new_price / self.price)
            self.return_history.append(ret)
            self.price = new_price
            self.history.append(self.price)
        self.spread_history.append(self.lob.spread())
        self.timestamp += 1
        return trades

    @property
    def ath(self) -> float:
        return max(self.history)

    @property
    def atl(self) -> float:
        return min(self.history)

    @property
    def ath_nearness(self) -> float:
        return self.price / self.ath if self.ath > 0 else 1.0

    @property
    def rolling_volatility(self, window=20) -> float:
        r = self.return_history
        if len(r) < 2: return 0.0
        return float(np.std(r[-window:]))




class PnLLedger:
    """FIFO-based realised/unrealised PnL + net-worth timeline."""

    def __init__(self, initial_cash: float):
        self.initial_cash = initial_cash
        self.cash: float = initial_cash
        self.position: int = 0
        self.realised_pnl: float = 0.0
        self._long_lots: deque = deque()   # (cost_basis, volume)
        self.net_worth_history: List[float] = [initial_cash]
        self.trade_count: int = 0

    def record_buy(self, price: float, volume: int):
        self.cash -= price * volume
        self.position += volume
        self._long_lots.append((price, volume))
        self.trade_count += 1

    def record_sell(self, price: float, volume: int):
        self.cash += price * volume
        self.position -= volume
        remaining = volume
        while remaining > 0 and self._long_lots:
            lp, lv = self._long_lots[0]
            matched = min(remaining, lv)
            self.realised_pnl += matched * (price - lp)
            remaining -= matched
            if matched == lv: self._long_lots.popleft()
            else: self._long_lots[0] = (lp, lv - matched)
        if remaining > 0:
            self.realised_pnl += remaining * price
        self.trade_count += 1

    def unrealised_pnl(self, market_price: float) -> float:
        if not self._long_lots: return 0.0
        avg = sum(p*v for p,v in self._long_lots) / sum(v for _,v in self._long_lots)
        return self.position * (market_price - avg)

    def net_worth(self, market_price: float) -> float:
        return self.cash + self.position * market_price

    def snapshot(self, market_price: float):
        self.net_worth_history.append(self.net_worth(market_price))

    def sharpe(self) -> float:
        h = np.array(self.net_worth_history)
        if len(h) < 2: return 0.0
        r = np.diff(h) / (h[:-1] + 1e-8)
        return float(np.mean(r) / (np.std(r) + 1e-8)) * np.sqrt(252)

    def max_drawdown(self) -> float:
        h = np.array(self.net_worth_history)
        peak = np.maximum.accumulate(h)
        dd = (h - peak) / (peak + 1e-8)
        return float(np.min(dd))

    def summary(self, market_price: float) -> Dict:
        return {
            'cash': round(self.cash, 2),
            'position': self.position,
            'realised_pnl': round(self.realised_pnl, 2),
            'unrealised_pnl': round(self.unrealised_pnl(market_price), 2),
            'net_worth': round(self.net_worth(market_price), 2),
            'sharpe': round(self.sharpe(), 3),
            'max_drawdown': round(self.max_drawdown() * 100, 2),
            'n_trades': self.trade_count,
        }




class LLMCache:
    """SHA-256 keyed in-memory cache for LLM completions."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._store: Dict[str, int] = {}
        self.hits = 0; self.misses = 0

    @staticmethod
    def _hash(ctx: Dict) -> str:
        return hashlib.sha256(json.dumps(ctx, sort_keys=True).encode()).hexdigest()

    def get(self, ctx: Dict) -> Optional[int]:
        if not self.enabled: return None
        v = self._store.get(self._hash(ctx))
        if v is not None: self.hits += 1
        else: self.misses += 1
        return v

    def set(self, ctx: Dict, direction: int):
        if self.enabled:
            self._store[self._hash(ctx)] = direction

    @property
    def hit_rate(self):
        t = self.hits + self.misses
        return self.hits / t if t else 0.0

    def stats(self):
        return {'hits': self.hits, 'misses': self.misses,
                'hit_rate': f'{self.hit_rate:.1%}',
                'cached': len(self._store)}


def ollama_trade_decision(context: Dict, model: str, cache: LLMCache) -> int:
    """
    Query Ollama for a BUY/SELL decision.
    Returns +1 (BUY) or -1 (SELL). Falls back to +1 if Ollama unavailable.
    """
    rounded = {
        'price': round(context['price'], 1),
        'ath': round(context['ath'], 1),
        'atl': round(context['atl'], 1),
        'ath_nearness': round(context['ath_nearness'], 3),
        'position': context['position'],
        'cash_bucket': int(context['cash'] // 500),
        'unrealised_pnl_sign': int(np.sign(context.get('unrealised_pnl', 0))),
    }
    cached = cache.get(rounded)
    if cached is not None:
        return cached

    prompt = (
        f'You are a stock trader.\n'
        f'Current price: {context["price"]:.2f}\n'
        f'All-time high: {context["ath"]:.2f} (nearness: {context["ath_nearness"]:.3f})\n'
        f'All-time low: {context["atl"]:.2f}\n'
        f'Position (shares held): {context["position"]}\n'
        f'Cash: {context["cash"]:.2f}\n'
        f'Unrealised PnL: {context.get("unrealised_pnl", 0):.2f}\n'
        f'Order flow imbalance: {context.get("ofi", 0):.3f}\n'
        f'Reply with exactly one word: BUY or SELL'
    )
    direction = 1  # default: buy
    try:
        import ollama
        resp = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
        text = resp['message']['content'].strip().upper()
        direction = -1 if 'SELL' in text else 1
    except Exception:
        direction = 1
    cache.set(rounded, direction)
    return direction




class FCNAgent:
    """Fundamental + Chartist + Noise agent (Chiarella & Iori 2002)."""

    def __init__(self, agent_id: int, cfg: SimConfig):
        self.id = agent_id
        self.cfg = cfg
        self.fundamental: float = cfg.fundamental_price
        self.wf = np.random.exponential(cfg.lambda_f)
        self.wc = np.random.exponential(cfg.lambda_c)
        self.wn = np.random.exponential(cfg.lambda_n)
        self.horizon: int = random.randint(cfg.horizon_min, cfg.horizon_max)
        self.ledger = PnLLedger(initial_cash=cfg.initial_cash)

    @property
    def position(self): return self.ledger.position
    @property
    def cash(self): return self.ledger.cash

    def predict_return(self, market: Market) -> float:
        p = market.price
        hist = market.history
        ptau = hist[-self.horizon] if len(hist) >= self.horizon else hist[0]
        w = self.wf + self.wc + self.wn
        r = (
            (self.wf / 200) * np.log(self.fundamental / (p + 1e-8))
            + (self.wc / self.horizon) * np.log((p + 1e-8) / (ptau + 1e-8))
            + self.wn * np.random.normal(0, 0.01)
        ) / w
        # ✅ FIX: clip r so the predicted price stays within ±10% of the
        # current market price: pred = p*exp(horizon*r) ∈ [0.90p, 1.10p].
        # Without this the chartist component creates a momentum cascade —
        # a small initial price move is amplified by high-wc agents whose
        # bids and asks drift to extreme prices, producing LOB midpoint
        # trades that explode over hundreds of steps.
        max_r = np.log(1.10) / max(self.horizon, 1)   # ≈ 0.095 / horizon
        return float(np.clip(r, -max_r, max_r))

    # ── position limits ──────────────────────────────────────────
    MAX_LONG  =  20   # max shares held long
    MAX_SHORT = -10   # max shares held short (allows short-selling)
    # ─────────────────────────────────────────────────────────────

    def act(self, market: Market):
        r = self.predict_return(market)
        pred = market.price * np.exp(self.horizon * r)
        side = 'buy' if pred > market.price else 'sell'
        # ✅ FIX: allow short-selling (position can go negative) so the LOB
        # is populated on BOTH sides from tick 1.  The original Chiarella &
        # Iori (2002) model has no position guard.  We add a soft limit
        # (MAX_SHORT / MAX_LONG) to prevent runaway exposure.
        if side == 'sell' and self.position <= self.MAX_SHORT: return
        if side == 'buy'  and self.position >= self.MAX_LONG:  return
        margin = np.random.uniform(0, 0.01)
        price = pred * (1 - margin) if side == 'buy' else pred * (1 + margin)
        market.submit_order(Order(str(uuid.uuid4()), self.id, side, float(price), 1, market.timestamp))


class TrendFollowerAgent(FCNAgent):
    """
    Exponential Moving Average crossover strategy.
    BUY when fast EMA > slow EMA; SELL when fast < slow.
    """
    def __init__(self, agent_id: int, cfg: SimConfig, fast: int = 10, slow: int = 40):
        super().__init__(agent_id, cfg)
        self.fast = fast
        self.slow = slow

    def act(self, market: Market):
        hist = market.history
        if len(hist) < self.slow + 1:
            return
        prices = np.array(hist[-(self.slow+1):])
        ema_fast = prices[-self.fast:].mean()
        ema_slow = prices.mean()
        side = 'buy' if ema_fast > ema_slow else 'sell'
        # ✅ FIX: same short-selling permission as FCNAgent
        if side == 'sell' and self.position <= self.MAX_SHORT: return
        if side == 'buy'  and self.position >= self.MAX_LONG:  return
        noise = np.random.normal(0, 0.005)
        price = market.price * (1 + noise)
        market.submit_order(Order(str(uuid.uuid4()), self.id, side, float(price), 1, market.timestamp))


class MeanReversionAgent(FCNAgent):
    """
    Mean-reversion via z-score.
    BUY when price is far below rolling mean; SELL when far above.
    """
    def __init__(self, agent_id: int, cfg: SimConfig, window: int = 30, z_thresh: float = 1.5):
        super().__init__(agent_id, cfg)
        self.window = window
        self.z_thresh = z_thresh

    def act(self, market: Market):
        hist = market.history
        if len(hist) < self.window + 1: return
        w = np.array(hist[-self.window:])
        mu, sig = w.mean(), w.std() + 1e-8
        z = (market.price - mu) / sig
        if z < -self.z_thresh:
            side = 'buy'
        elif z > self.z_thresh:
            side = 'sell'
        else:
            return  # no signal
        # ✅ FIX: same short-selling permission as FCNAgent
        if side == 'sell' and self.position <= self.MAX_SHORT: return
        if side == 'buy'  and self.position >= self.MAX_LONG:  return
        margin = np.random.uniform(0, 0.005)
        price = market.price * (1 - margin) if side == 'buy' else market.price * (1 + margin)
        market.submit_order(Order(str(uuid.uuid4()), self.id, side, float(price), 1, market.timestamp))


class FCLAgent(FCNAgent):
    """FCN + LLM buy/sell direction (from paper). Graceful fallback if Ollama is absent."""

    def __init__(self, agent_id: int, cfg: SimConfig, cache: LLMCache):
        super().__init__(agent_id, cfg)
        self._cache = cache

    def act(self, market: Market):
        context = {
            'price': market.price,
            'ath': market.ath,
            'atl': market.atl,
            'ath_nearness': market.ath_nearness,
            'position': self.position,
            'cash': self.cash,
            'unrealised_pnl': self.ledger.unrealised_pnl(market.price),
            'ofi': market.lob.ofi(),
        }
        direction = ollama_trade_decision(context, self.cfg.llm_model, self._cache)
        side = 'buy' if direction > 0 else 'sell'
        # ✅ FIX: same short-selling permission as FCNAgent
        if side == 'sell' and self.position <= self.MAX_SHORT: return
        if side == 'buy'  and self.position >= self.MAX_LONG:  return
        r = self.predict_return(market)
        pred = market.price * np.exp(self.horizon * r)
        margin = np.random.uniform(0, 0.01)
        price = pred * (1 - margin) if side == 'buy' else pred * (1 + margin)
        market.submit_order(Order(str(uuid.uuid4()), self.id, side, float(price), 1, market.timestamp))




N_STATE = 8
N_ACTIONS = 3   # 0=hold, 1=buy, 2=sell


def encode_state(market: Market, ledger: PnLLedger, initial_cash: float) -> np.ndarray:
    """Encode market + portfolio into a normalized numpy feature vector."""
    hist = market.history
    p = market.price

    ath = max(hist) if hist else p
    atl = min(hist) if hist else p
    ath_near = np.clip(p / (ath + 1e-8), 0, 2)
    atl_near = np.clip(p / (atl + 1e-8), 0, 2)

    p10 = hist[-10] if len(hist) >= 10 else hist[0]
    p50 = hist[-50] if len(hist) >= 50 else hist[0]
    short_mom = np.clip(np.log((p + 1e-8) / (p10 + 1e-8)) / 0.1, -3, 3)
    long_mom = np.clip(np.log((p + 1e-8) / (p50 + 1e-8)) / 0.2, -3, 3)

    if len(market.return_history) >= 5:
        vol = np.clip(np.std(market.return_history[-20:]) / 0.05, 0, 5)
    else:
        vol = 0.0

    max_pos = max(initial_cash / (p + 1e-8), 1.0)
    pos_norm = np.clip(ledger.position / max_pos, -2, 2)
    cash_ratio = np.clip(ledger.cash / (initial_cash + 1e-8), -1, 3)
    unreal_ratio = np.clip(ledger.unrealised_pnl(p) / (initial_cash + 1e-8), -1, 1)

    return np.array([ath_near, atl_near, short_mom, long_mom,
                     vol, pos_norm, cash_ratio, unreal_ratio], dtype=np.float32)


def discretize_state(state: np.ndarray, bins: int = 5) -> int:
    """
    Compact tabular encoding using first 4 state features.
    Total states = bins^4 = 625 for bins=5.
    """
    feat = np.array([
        np.clip(state[0], 0.5, 1.5),   # ATH nearness → [0.5, 1.5]
        np.clip(state[2], -2, 2),       # short momentum
        np.clip(state[5], -1.5, 1.5),   # position
        np.clip(state[6], 0, 2.5),      # cash ratio
    ])
    lo = np.array([0.5, -2, -1.5, 0])
    hi = np.array([1.5, 2,   1.5, 2.5])
    bucket = np.floor((feat - lo) / (hi - lo + 1e-8) * bins).astype(int)
    bucket = np.clip(bucket, 0, bins - 1)
    idx = 0
    for b in bucket:
        idx = idx * bins + b
    return idx


class ReplayBuffer:
    """Fixed-capacity circular experience replay for DQN."""

    def __init__(self, capacity: int = 20_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, float(done)))

    def sample(self, batch: int):
        idxs = np.random.choice(len(self.buf), batch, replace=False)
        s, a, r, s2, d = zip(*[self.buf[i] for i in idxs])
        return (np.array(s, dtype=np.float32), np.array(a),
                np.array(r, dtype=np.float32), np.array(s2, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self): return len(self.buf)




class NumpyMLP:
    """
    Pure-NumPy Multi-Layer Perceptron.
    Architecture: [input] → [hidden...] → [output], ReLU hidden, linear output.
    Training: SGD with gradient clipping + He initialization.
    No PyTorch or TensorFlow required.
    """

    def __init__(self, layer_sizes: List[int], lr: float = 5e-4):
        self.lr = lr
        self.W: List[np.ndarray] = []
        self.b: List[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            self.W.append(np.random.randn(fan_in, layer_sizes[i+1]) * np.sqrt(2.0 / fan_in))
            self.b.append(np.zeros(layer_sizes[i+1]))

    def _forward(self, x: np.ndarray):
        """Return (output, list of (pre_activation, activation)) for backprop."""
        cache = []
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(0, z) if i < len(self.W) - 1 else z
            cache.append((z, h))
        return h, cache

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = x.reshape(-1, self.W[0].shape[0]) if x.ndim == 1 else x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.maximum(0, z) if i < len(self.W) - 1 else z
        return h

    def train_step(self, x: np.ndarray, target: np.ndarray) -> float:
        """One SGD step on a batch. Returns MSE loss."""
        B = x.shape[0]
        # Forward
        acts = [x]
        pres = []
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            pres.append(z)
            h = np.maximum(0, z) if i < len(self.W) - 1 else z
            acts.append(h)
        # Loss
        loss = np.mean((acts[-1] - target) ** 2)
        # Backward
        delta = 2 * (acts[-1] - target) / B
        for i in reversed(range(len(self.W))):
            dW = np.clip(acts[i].T @ delta, -1.0, 1.0)
            db = np.clip(delta.sum(0), -1.0, 1.0)
            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * db
            if i > 0:
                delta = (delta @ self.W[i].T) * (pres[i-1] > 0).astype(float)
        return float(loss)

    def copy_from(self, other: 'NumpyMLP'):
        """Hard-copy weights (for target network sync)."""
        for i in range(len(self.W)):
            self.W[i] = other.W[i].copy()
            self.b[i] = other.b[i].copy()


# Quick sanity check
net = NumpyMLP([N_STATE, 64, 64, N_ACTIONS], lr=1e-3)
x_test = np.random.randn(32, N_STATE).astype(np.float32)
y_test = np.random.randn(32, N_ACTIONS).astype(np.float32)
l0 = net.train_step(x_test, y_test)


class QLearningAgent(FCNAgent):
    """
    Tabular Q-Learning Agent.

    Algorithm
    ---------
    - State: discretized 4-feature vector → integer index
    - Policy: ε-greedy (decaying)
    - Update: Q(s,a) ← Q(s,a) + α[r + γ max_a Q(s',a) - Q(s,a)]
    - Reward: Δ net_worth (normalised by initial_cash)

    Preserves Q-table across episodes (persistent learning).
    """

    N_DISC_STATES = 5 ** 4   # 625

    def __init__(self, agent_id: int, cfg: SimConfig,
                 alpha: float = None, gamma: float = None,
                 epsilon: float = None, epsilon_decay: float = None,
                 epsilon_min: float = None):
        super().__init__(agent_id, cfg)
        self.alpha = alpha or cfg.rl_alpha
        self.gamma = gamma or cfg.rl_gamma
        self.epsilon = epsilon if epsilon is not None else cfg.rl_epsilon_start
        self.epsilon_decay = epsilon_decay or cfg.rl_epsilon_decay
        self.epsilon_min = epsilon_min or cfg.rl_epsilon_min

        # Q-table: rows = states, cols = actions
        self.Q = np.zeros((self.N_DISC_STATES, N_ACTIONS))

        self._s = None     # previous state index
        self._a = None     # previous action
        self._prev_worth = cfg.initial_cash

        # Diagnostics
        self.episode_rewards: List[float] = []
        self.td_errors: List[float] = []
        self.n_steps = 0

    def _state_idx(self, market: Market) -> int:
        s = encode_state(market, self.ledger, self.cfg.initial_cash)
        return discretize_state(s)

    def _choose(self, idx: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        return int(np.argmax(self.Q[idx]))

    RL_MAX_LONG  =  15
    RL_MAX_SHORT = -5

    def _execute(self, action: int, market: Market):
        p = market.price
        r = self.predict_return(market)
        pred = p * np.exp(self.horizon * r)
        # ✅ FIX: sell allowed when position > RL_MAX_SHORT (short-selling enabled)
        if action == 1 and self.position < self.RL_MAX_LONG:    # buy
            pr = pred * (1 - np.random.uniform(0, 0.01))
            market.submit_order(Order(str(uuid.uuid4()), self.id, 'buy', float(pr), 1, market.timestamp))
        elif action == 2 and self.position > self.RL_MAX_SHORT:  # sell (short OK)
            pr = pred * (1 + np.random.uniform(0, 0.01))
            market.submit_order(Order(str(uuid.uuid4()), self.id, 'sell', float(pr), 1, market.timestamp))
        # action 0 = hold: no order

    def act(self, market: Market):
        s_idx = self._state_idx(market)
        action = self._choose(s_idx)

        # Q-update from previous transition
        if self._s is not None:
            worth = self.ledger.net_worth(market.price)
            reward = (worth - self._prev_worth) / (self.cfg.initial_cash + 1e-8)
            td_target = reward + self.gamma * np.max(self.Q[s_idx])
            td_err = td_target - self.Q[self._s, self._a]
            self.Q[self._s, self._a] += self.alpha * td_err
            self.episode_rewards.append(reward * self.cfg.initial_cash)
            self.td_errors.append(abs(td_err))

        self._s = s_idx
        self._a = action
        self._prev_worth = self.ledger.net_worth(market.price)

        self._execute(action, market)

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.n_steps += 1

    def reset_episode(self):
        """Reset per-episode state without touching Q-table."""
        self._s = None; self._a = None
        self._prev_worth = self.cfg.initial_cash
        self.ledger = PnLLedger(initial_cash=self.cfg.initial_cash)


      f'{QLearningAgent.N_DISC_STATES * N_ACTIONS} parameters')


class DQNAgent(FCNAgent):
    """
    Double DQN Agent with Experience Replay — pure NumPy.

    Innovations vs vanilla DQN
    --------------------------
    1. **Double DQN**: online network selects action; target network evaluates value.
       Reduces overestimation bias.
    2. **Target network**: hard-copied every `target_update_freq` steps.
    3. **Experience replay**: uniform sampling from a 20k circular buffer.
    4. **Reward normalisation**: rewards scaled by initial_cash for training stability.
    5. **Feasibility mask**: infeasible actions (buy with no cash, sell with no stock)
       receive a large negative Q-value before action selection.
    """

    def __init__(self, agent_id: int, cfg: SimConfig,
                 hidden: List[int] = None,
                 lr: float = None, gamma: float = None,
                 epsilon: float = None, epsilon_decay: float = None,
                 epsilon_min: float = None):
        super().__init__(agent_id, cfg)
        hidden = hidden or [64, 64]
        arch = [N_STATE] + hidden + [N_ACTIONS]
        self.lr = lr or cfg.dqn_lr
        self.gamma = gamma or cfg.rl_gamma
        self.epsilon = epsilon if epsilon is not None else cfg.rl_epsilon_start
        self.epsilon_decay = epsilon_decay or cfg.rl_epsilon_decay
        self.epsilon_min = epsilon_min or cfg.rl_epsilon_min
        self.batch = cfg.dqn_batch
        self.target_freq = cfg.dqn_target_update

        self.online = NumpyMLP(arch, lr=self.lr)
        self.target = NumpyMLP(arch, lr=self.lr)
        self.target.copy_from(self.online)

        self.buffer = ReplayBuffer(cfg.replay_capacity)

        self._prev_s: Optional[np.ndarray] = None
        self._prev_a: Optional[int] = None
        self._prev_worth: float = cfg.initial_cash

        # Diagnostics
        self.losses: List[float] = []
        self.episode_rewards: List[float] = []
        self.n_steps = 0

    def _get_state(self, market: Market) -> np.ndarray:
        return encode_state(market, self.ledger, self.cfg.initial_cash)

    RL_MAX_LONG  =  15   # RL agent position limits (tighter than FCN)
    RL_MAX_SHORT = -5

    def _feasible_q(self, q: np.ndarray, market: Market) -> np.ndarray:
        """Mask infeasible actions with a large penalty."""
        q = q.copy()
        # ✅ FIX: allow short-selling for RL agent (down to RL_MAX_SHORT)
        if self.position >= self.RL_MAX_LONG:
            q[1] = -1e9  # already at long limit
        if self.position <= self.RL_MAX_SHORT:
            q[2] = -1e9  # already at short limit
        return q

    def _choose(self, s: np.ndarray, market: Market) -> int:
        if np.random.random() < self.epsilon:
            # ✅ FIX: sell is feasible whenever position > RL_MAX_SHORT
            choices = [0]  # hold always feasible
            if self.position < self.RL_MAX_LONG:  choices.append(1)  # buy
            if self.position > self.RL_MAX_SHORT: choices.append(2)  # sell (short OK)
            return random.choice(choices)
        q = self.online.predict(s.reshape(1, -1))[0]
        q = self._feasible_q(q, market)
        return int(np.argmax(q))

    def _execute(self, action: int, market: Market):
        p = market.price
        r = self.predict_return(market)
        # ✅ FIX: sell allowed when position > RL_MAX_SHORT (short-selling enabled)
        if action == 1 and self.position < self.RL_MAX_LONG:
            price = p * np.exp(self.horizon * r) * (1 - np.random.uniform(0, 0.01))
            market.submit_order(Order(str(uuid.uuid4()), self.id, 'buy', float(price), 1, market.timestamp))
        elif action == 2 and self.position > self.RL_MAX_SHORT:
            price = p * np.exp(self.horizon * r) * (1 + np.random.uniform(0, 0.01))
            market.submit_order(Order(str(uuid.uuid4()), self.id, 'sell', float(price), 1, market.timestamp))

    def _train(self):
        if len(self.buffer) < self.batch: return None
        S, A, R, S2, D = self.buffer.sample(self.batch)
        # Double DQN targets
        q_curr = self.online.predict(S)
        q_next_online = self.online.predict(S2)
        q_next_target = self.target.predict(S2)
        best_a = np.argmax(q_next_online, axis=1)       # online selects action
        next_v = q_next_target[np.arange(self.batch), best_a]  # target evaluates
        # Build TD targets (only update selected actions)
        tgt = q_curr.copy()
        for i in range(self.batch):
            tgt[i, A[i]] = R[i] + self.gamma * next_v[i] * (1 - D[i])
        loss = self.online.train_step(S, tgt)
        return loss

    def act(self, market: Market):
        s = self._get_state(market)
        action = self._choose(s, market)

        if self._prev_s is not None:
            worth = self.ledger.net_worth(market.price)
            reward = (worth - self._prev_worth) / (self.cfg.initial_cash + 1e-8)
            self.buffer.push(self._prev_s, self._prev_a, reward, s, False)
            self.episode_rewards.append(reward * self.cfg.initial_cash)
            loss = self._train()
            if loss is not None: self.losses.append(loss)
            if self.n_steps % self.target_freq == 0:
                self.target.copy_from(self.online)

        self._prev_s = s
        self._prev_a = action
        self._prev_worth = self.ledger.net_worth(market.price)

        self._execute(action, market)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.n_steps += 1

    def reset_episode(self):
        """Keep network weights + replay buffer; reset portfolio."""
        self._prev_s = None; self._prev_a = None
        self._prev_worth = self.cfg.initial_cash
        self.ledger = PnLLedger(initial_cash=self.cfg.initial_cash)


dqn_test = DQNAgent(0, DEFAULT_CFG)
params = sum(W.size + b.size for W, b in zip(dqn_test.online.W, dqn_test.online.b))


def build_agents(cfg: SimConfig, llm_cache: LLMCache,
                 extra_agents: List = None,
                 n_trend: int = 5, n_mean_rev: int = 5) -> List:
    """
    Construct a heterogeneous agent pool:
    - FCNAgents (bulk of the market)
    - TrendFollowerAgents
    - MeanReversionAgents
    - FCLAgents (optional, needs Ollama)
    - extra_agents: pre-built RL agents passed in
    """
    n_extra = len(extra_agents) if extra_agents else 0
    n_fcn = cfg.n_agents - cfg.n_llm - n_trend - n_mean_rev - n_extra
    n_fcn = max(n_fcn, 0)
    agents = [FCNAgent(i, cfg) for i in range(n_fcn)]
    base = n_fcn
    agents += [TrendFollowerAgent(base + i, cfg,
                                  fast=random.randint(5, 15),
                                  slow=random.randint(30, 60))
               for i in range(n_trend)]
    base += n_trend
    agents += [MeanReversionAgent(base + i, cfg,
                                  window=random.randint(20, 50),
                                  z_thresh=np.random.uniform(1.0, 2.5))
               for i in range(n_mean_rev)]
    base += n_mean_rev
    if cfg.n_llm > 0:
        agents += [FCLAgent(base + i, cfg, llm_cache) for i in range(cfg.n_llm)]
        base += cfg.n_llm
    if extra_agents:
        for a in extra_agents:
            # ✅ FIX: Do NOT reassign a.id here.
            # Extra agents (RL) are pre-built with stable ids (e.g. 900, 901).
            # Overwriting their id breaks any dict keyed on the original id
            # (e.g. TrainingLoop.ep_final_worth).  Regular agents use ids 0–N,
            # which never collide with ids ≥ 900, so no dedup is needed.
            agents.append(a)
    return agents


class Simulation:
    """
    Core simulation engine.
    Supports plug-in stochastic fundamental processes and RL agents.
    """

    def __init__(self, cfg: SimConfig,
                 process: StochasticProcess = None,
                 extra_agents: List = None,
                 llm_cache: LLMCache = None,
                 n_trend: int = 5, n_mean_rev: int = 5):
        self.cfg = cfg
        cfg.seed_everything()
        self.process = process or GBMProcess(cfg.fundamental_price, sigma=1e-3)
        self.market = Market(cfg)
        self.llm_cache = llm_cache or LLMCache(enabled=cfg.llm_cache_enabled)
        self.agents = build_agents(cfg, self.llm_cache, extra_agents, n_trend, n_mean_rev)
        self._amap: Dict[int, FCNAgent] = {a.id: a for a in self.agents}

    def run(self, steps: int = 500, verbose: bool = True) -> 'SimResult':
        it = tqdm(range(steps), desc=f'seed={self.cfg.seed}', leave=False) if verbose else range(steps)
        for _ in it:
            # Step fundamental process
            new_fund = self.process.step()
            for a in self.agents:
                a.fundamental = new_fund
            # Agent acts
            agent = random.choice(self.agents)
            agent.act(self.market)
            trades = self.market.step()
            self._settle(trades)
            self._snapshot()
        return SimResult(self)

    def _settle(self, trades):
        for tp, vol, bid_id, ask_id in trades:
            b = self._amap.get(bid_id)
            s = self._amap.get(ask_id)
            if b: b.ledger.record_buy(tp, vol)
            if s: s.ledger.record_sell(tp, vol)

    def _snapshot(self):
        p = self.market.price
        for a in self.agents: a.ledger.snapshot(p)


class SimResult:
    """Structured output of a completed simulation."""

    def __init__(self, sim: Simulation):
        p = sim.market.price
        self.prices = list(sim.market.history)
        self.returns = list(sim.market.return_history)
        self.spread_history = sim.market.spread_history
        self.fundamental_path = list(sim.process.prices)
        self.final_price = p
        self.agents = sim.agents
        self.cfg = sim.cfg
        self.llm_cache_stats = sim.llm_cache.stats()

        # Wealth timelines keyed by agent_id
        self.wealth_timelines: Dict[int, List[float]] = {
            a.id: a.ledger.net_worth_history for a in sim.agents
        }

    def perf_table(self) -> pd.DataFrame:
        rows = []
        p = self.final_price
        ic = self.cfg.initial_cash
        for a in self.agents:
            nw = a.ledger.net_worth(p)
            rows.append({
                'agent_id': a.id,
                'type': type(a).__name__,
                'net_worth': round(nw, 2),
                'total_return_%': round((nw / ic - 1) * 100, 2),
                'realised_pnl': round(a.ledger.realised_pnl, 2),
                'unrealised_pnl': round(a.ledger.unrealised_pnl(p), 2),
                'position': a.ledger.position,
                'sharpe': round(a.ledger.sharpe(), 3),
                'max_dd_%': round(a.ledger.max_drawdown() * 100, 2),
                'n_trades': a.ledger.trade_count,
            })
        return pd.DataFrame(rows).sort_values('net_worth', ascending=False).reset_index(drop=True)



