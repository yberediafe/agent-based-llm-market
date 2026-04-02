"""
Core unit and integration tests for abm_market_sim.

Run with:
    pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import random
import numpy as np
import pytest

from abm_market_sim import (
    SimConfig, Order, LimitOrderBook, Market, PnLLedger,
    GBMProcess, OUProcess, MertonJumpDiffusion, HestonProcess,
    FCNAgent, TrendFollowerAgent, MeanReversionAgent,
    QLearningAgent, DQNAgent,
    Simulation, SimResult,
    encode_state, discretize_state,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_cfg(**kw) -> SimConfig:
    defaults = dict(seed=0, n_agents=20, n_llm=0, initial_cash=10_000.0)
    defaults.update(kw)
    return SimConfig(**defaults)


def run_sim(steps=200, n_agents=30, n_trend=2, n_mean_rev=2,
            process=None, extra_agents=None, seed=0) -> SimResult:
    random.seed(seed); np.random.seed(seed)
    cfg = make_cfg(seed=seed, n_agents=n_agents)
    proc = process or GBMProcess(300.0, sigma=1e-3)
    sim = Simulation(cfg, process=proc, n_trend=n_trend,
                     n_mean_rev=n_mean_rev, extra_agents=extra_agents)
    return sim.run(steps=steps, verbose=False)


# ─────────────────────────────────────────────────────────────────────────────
# Stochastic processes
# ─────────────────────────────────────────────────────────────────────────────

class TestStochasticProcesses:
    def test_gbm_positive(self):
        np.random.seed(1)
        proc = GBMProcess(300.0, sigma=1e-3)
        for _ in range(200): proc.step()
        assert all(p > 0 for p in proc.prices)

    def test_gbm_length(self):
        proc = GBMProcess(300.0)
        for _ in range(50): proc.step()
        assert len(proc.prices) == 51  # initial + 50 steps

    def test_ou_mean_reversion(self):
        np.random.seed(2)
        proc = OUProcess(300.0, kappa=0.5, theta=300.0, sigma=0.5)
        for _ in range(1000): proc.step()
        prices = np.array(proc.prices[100:])
        # With strong kappa, mean should stay close to theta
        assert abs(prices.mean() - 300.0) < 30.0

    def test_merton_fat_tails(self):
        np.random.seed(3)
        proc = MertonJumpDiffusion(300.0, lam=0.1, mu_j=-0.05, sigma_j=0.05)
        for _ in range(500): proc.step()
        assert all(p > 0 for p in proc.prices)

    def test_heston_positive_variance(self):
        np.random.seed(4)
        proc = HestonProcess(300.0, v0=1e-4, kappa=2.0, theta=1e-4, xi=0.01)
        for _ in range(300): proc.step()
        assert all(p > 0 for p in proc.prices)
        assert all(v >= 0 for v in proc.vol_history)

    def test_reset(self):
        proc = GBMProcess(300.0)
        for _ in range(10): proc.step()
        proc.reset()
        assert proc.current == 300.0
        assert len(proc.prices) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Limit Order Book
# ─────────────────────────────────────────────────────────────────────────────

class TestLimitOrderBook:
    def make_order(self, side, price, agent_id=0, volume=1):
        import uuid
        return Order(str(uuid.uuid4()), agent_id, side, price, volume, 0)

    def test_no_match_when_spread_positive(self):
        lob = LimitOrderBook()
        lob.add_order(self.make_order('buy', 100.0, agent_id=1))
        lob.add_order(self.make_order('sell', 110.0, agent_id=2))
        trades = lob.match(0)
        assert trades == []

    def test_match_when_bid_geq_ask(self):
        lob = LimitOrderBook()
        lob.add_order(self.make_order('buy', 110.0, agent_id=1))
        lob.add_order(self.make_order('sell', 100.0, agent_id=2))
        trades = lob.match(0)
        assert len(trades) == 1
        price, vol, bid_id, ask_id = trades[0]
        assert price == pytest.approx(105.0)
        assert vol == 1

    def test_trade_price_is_midpoint(self):
        lob = LimitOrderBook()
        lob.add_order(self.make_order('buy', 120.0, agent_id=1))
        lob.add_order(self.make_order('sell', 80.0, agent_id=2))
        trades = lob.match(0)
        assert trades[0][0] == pytest.approx(100.0)

    def test_cancel_removes_order(self):
        lob = LimitOrderBook()
        import uuid
        oid = str(uuid.uuid4())
        o = Order(oid, 0, 'buy', 100.0, 1, 0)
        lob.add_order(o)
        assert lob.cancel_order(oid)
        assert len(lob.bids) == 0

    def test_spread(self):
        lob = LimitOrderBook()
        lob.add_order(self.make_order('buy', 99.0, agent_id=1))
        lob.add_order(self.make_order('sell', 101.0, agent_id=2))
        assert lob.spread() == pytest.approx(2.0)

    def test_ofi_symmetric(self):
        lob = LimitOrderBook()
        lob.add_order(self.make_order('buy', 99.0, agent_id=1))
        lob.add_order(self.make_order('sell', 101.0, agent_id=2))
        assert abs(lob.ofi()) < 1e-9  # equal volume on both sides


# ─────────────────────────────────────────────────────────────────────────────
# PnL Ledger
# ─────────────────────────────────────────────────────────────────────────────

class TestPnLLedger:
    def test_buy_reduces_cash(self):
        l = PnLLedger(10_000.0)
        l.record_buy(100.0, 5)
        assert l.cash == pytest.approx(9_500.0)
        assert l.position == 5

    def test_sell_increases_cash(self):
        l = PnLLedger(10_000.0)
        l.record_buy(100.0, 5)
        l.record_sell(120.0, 5)
        assert l.cash == pytest.approx(9_500.0 + 600.0)
        assert l.position == 0

    def test_realised_pnl_fifo(self):
        l = PnLLedger(10_000.0)
        l.record_buy(100.0, 3)
        l.record_buy(110.0, 2)
        l.record_sell(120.0, 3)  # closes 3 at cost 100 → PnL = 3*(120-100) = 60
        assert l.realised_pnl == pytest.approx(60.0)

    def test_net_worth(self):
        l = PnLLedger(10_000.0)
        l.record_buy(100.0, 10)
        assert l.net_worth(100.0) == pytest.approx(10_000.0)
        assert l.net_worth(110.0) == pytest.approx(10_100.0)

    def test_snapshot_history(self):
        l = PnLLedger(10_000.0)
        l.snapshot(300.0)
        l.snapshot(310.0)
        assert len(l.net_worth_history) == 3  # initial + 2 snapshots

    def test_short_position_realised(self):
        l = PnLLedger(10_000.0)
        l.record_sell(100.0, 3)   # short-sell 3 shares
        assert l.position == -3
        assert l.cash == pytest.approx(10_300.0)

    def test_max_drawdown_negative(self):
        l = PnLLedger(10_000.0)
        for v in [10_000, 9_000, 8_000, 9_500, 10_200]:
            l.net_worth_history.append(v)
        dd = l.max_drawdown()
        assert dd < 0


# ─────────────────────────────────────────────────────────────────────────────
# Market
# ─────────────────────────────────────────────────────────────────────────────

class TestMarket:
    def test_price_updates_on_trade(self):
        import uuid
        cfg = make_cfg()
        m = Market(cfg)
        m.submit_order(Order(str(uuid.uuid4()), 1, 'buy', 310.0, 1, 0))
        m.submit_order(Order(str(uuid.uuid4()), 2, 'sell', 290.0, 1, 0))
        m.step()
        assert m.price == pytest.approx(300.0)

    def test_ath_atl(self):
        import uuid
        cfg = make_cfg()
        m = Market(cfg)
        m.submit_order(Order(str(uuid.uuid4()), 1, 'buy', 350.0, 1, 0))
        m.submit_order(Order(str(uuid.uuid4()), 2, 'sell', 290.0, 1, 0))
        m.step()
        assert m.ath == pytest.approx(320.0)  # midpoint
        assert m.atl == pytest.approx(300.0)  # initial still lower


# ─────────────────────────────────────────────────────────────────────────────
# State encoder
# ─────────────────────────────────────────────────────────────────────────────

class TestStateEncoder:
    def test_shape(self):
        cfg = make_cfg()
        m = Market(cfg)
        ledger = PnLLedger(cfg.initial_cash)
        s = encode_state(m, ledger, cfg.initial_cash)
        assert s.shape == (8,)

    def test_discretize_in_range(self):
        cfg = make_cfg()
        m = Market(cfg)
        ledger = PnLLedger(cfg.initial_cash)
        s = encode_state(m, ledger, cfg.initial_cash)
        idx = discretize_state(s)
        assert 0 <= idx < 625

    def test_deterministic(self):
        cfg = make_cfg()
        m = Market(cfg)
        ledger = PnLLedger(cfg.initial_cash)
        s1 = discretize_state(encode_state(m, ledger, cfg.initial_cash))
        s2 = discretize_state(encode_state(m, ledger, cfg.initial_cash))
        assert s1 == s2


# ─────────────────────────────────────────────────────────────────────────────
# Full simulation integration
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulation:
    def test_trades_are_executed(self):
        result = run_sim(steps=200, n_agents=30, seed=1)
        total_trades = len([r for r in result.prices if r != 300.0])
        assert total_trades > 0, "No price movement — no trades executed"

    def test_price_moves(self):
        result = run_sim(steps=300, seed=2)
        assert result.final_price != 300.0

    def test_price_bounded(self):
        result = run_sim(steps=500, n_agents=60, seed=3)
        prices = result.prices
        assert max(prices) < 300.0 * 5,  f"Price too high: {max(prices):.1f}"
        assert min(prices) > 300.0 / 5,  f"Price too low: {min(prices):.1f}"

    def test_short_selling_occurs(self):
        result = run_sim(steps=400, n_agents=50, seed=4)
        positions = [a.ledger.position for a in result.agents]
        assert any(p < 0 for p in positions), "No short positions found"

    def test_position_limits_respected(self):
        result = run_sim(steps=500, n_agents=60, seed=5)
        for a in result.agents:
            assert a.ledger.position <= 20,  f"Agent {a.id} exceeded MAX_LONG"
            assert a.ledger.position >= -10, f"Agent {a.id} exceeded MAX_SHORT"

    def test_perf_table_shape(self):
        result = run_sim(steps=200, n_agents=20, seed=6)
        df = result.perf_table()
        assert len(df) == 20
        assert 'net_worth' in df.columns
        assert 'sharpe' in df.columns

    def test_four_processes(self):
        """Each stochastic process produces valid price history."""
        procs = [
            GBMProcess(300.0, sigma=1e-3),
            OUProcess(300.0, kappa=0.02, theta=300.0, sigma=0.8),
            MertonJumpDiffusion(300.0, lam=0.02),
            HestonProcess(300.0, v0=1e-4, kappa=2.0, theta=1e-4, xi=0.01),
        ]
        for i, proc in enumerate(procs):
            result = run_sim(steps=200, process=proc, seed=10 + i)
            assert result.final_price > 0, f"Process {type(proc).__name__} gave non-positive price"

    def test_rl_agents_in_market(self):
        random.seed(7); np.random.seed(7)
        cfg = make_cfg(seed=7, n_agents=40)
        ql  = QLearningAgent(agent_id=900, cfg=cfg)
        dqn = DQNAgent(agent_id=901, cfg=cfg)
        proc = GBMProcess(300.0, sigma=1e-3)
        sim  = Simulation(cfg, process=proc, extra_agents=[ql, dqn],
                          n_trend=2, n_mean_rev=2)
        result = sim.run(steps=200, verbose=False)
        # RL agents should still have positive net worth
        fp = result.final_price
        assert ql.ledger.net_worth(fp) > 0
        assert dqn.ledger.net_worth(fp) > 0

    def test_reproducibility(self):
        """Same seed → identical price history."""
        r1 = run_sim(steps=100, seed=42)
        r2 = run_sim(steps=100, seed=42)
        assert r1.prices == r2.prices

    def test_different_seeds_differ(self):
        r1 = run_sim(steps=100, seed=1)
        r2 = run_sim(steps=100, seed=2)
        assert r1.prices != r2.prices


# ─────────────────────────────────────────────────────────────────────────────
# NumpyMLP
# ─────────────────────────────────────────────────────────────────────────────

class TestNumpyMLP:
    def test_forward_shape(self):
        from abm_market_sim import NumpyMLP
        net = NumpyMLP([8, 32, 3])
        x = np.random.randn(16, 8).astype(np.float32)
        out = net.predict(x)
        assert out.shape == (16, 3)

    def test_loss_decreases(self):
        from abm_market_sim import NumpyMLP
        np.random.seed(0)
        net = NumpyMLP([8, 32, 3], lr=1e-2)
        x = np.random.randn(64, 8).astype(np.float32)
        y = np.random.randn(64, 3).astype(np.float32)
        l0 = net.train_step(x, y)
        for _ in range(50): net.train_step(x, y)
        l_final = net.train_step(x, y)
        assert l_final < l0, "Loss did not decrease after 50 steps"

    def test_copy_from(self):
        from abm_market_sim import NumpyMLP
        net1 = NumpyMLP([8, 16, 3])
        net2 = NumpyMLP([8, 16, 3])
        net2.copy_from(net1)
        x = np.random.randn(1, 8).astype(np.float32)
        np.testing.assert_array_equal(net1.predict(x), net2.predict(x))
