import sys
from pathlib import Path
import pytest

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bot import Config, TraderBot, SymbolFetcher


def test_init_requires_positive_trade_size():
    with pytest.raises(ValueError, match="non-positive trade size"):
        TraderBot(Config(stake_usd=0, risk_pct=0, max_tokens=float("inf")))


def test_init_requires_positive_max_tokens():
    with pytest.raises(ValueError, match="non-positive trade size"):
        TraderBot(Config(stake_usd=100, risk_pct=0.01, max_tokens=0))


def test_init_with_positive_stake(monkeypatch):
    monkeypatch.setattr(SymbolFetcher, "start", lambda self: None)
    TraderBot(Config(stake_usd=10, risk_pct=0))


def test_init_with_positive_risk_pct(monkeypatch):
    monkeypatch.setattr(SymbolFetcher, "start", lambda self: None)
    TraderBot(Config(stake_usd=0, risk_pct=0.01))
