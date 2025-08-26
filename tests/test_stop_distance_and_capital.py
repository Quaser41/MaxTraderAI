import sys
from pathlib import Path
import logging
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from bot import Config, TraderBot, SymbolFetcher


def test_execute_trade_skips_on_zero_stop_distance(monkeypatch, caplog):
    symbol = "TEST-USD"
    price = 100.0
    timestamp = pd.Timestamp("2024-01-01")
    monkeypatch.setattr(SymbolFetcher, "start", lambda self: None)
    config = Config(
        symbol=symbol,
        risk_pct=0.01,
        atr_multiplier=0,
        stop_loss_pct=0.01,
        spread_pct=-0.02,
    )
    bot = TraderBot(config)
    with caplog.at_level(logging.WARNING):
        bot.execute_trade("buy", price, timestamp, symbol)
    assert symbol not in bot.account.positions
    assert any("non-positive stop distance" in rec.message for rec in caplog.records)


def test_execute_trade_falls_back_when_exceeds_capital(monkeypatch, caplog):
    symbol = "TEST-USD"
    price = 10.0
    timestamp = pd.Timestamp("2024-01-01")
    monkeypatch.setattr(SymbolFetcher, "start", lambda self: None)
    config = Config(
        symbol=symbol,
        risk_pct=0.1,
        atr_multiplier=0,
        stop_loss_pct=0.01,
        stake_usd=100.0,
        max_exposure=0.5,
    )
    bot = TraderBot(config)
    with caplog.at_level(logging.INFO):
        bot.execute_trade("buy", price, timestamp, symbol)
    pos = bot.account.positions[symbol]
    assert pos["amount"] == pytest.approx(config.stake_usd / price)
    assert any("capital limit" in rec.message for rec in caplog.records)
