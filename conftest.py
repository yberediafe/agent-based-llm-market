"""
Pytest configuration — adds src/ to sys.path so tests can import abm_market_sim
without installing the package.
"""
import sys
import os

# Allow `from abm_market_sim import ...` in tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
