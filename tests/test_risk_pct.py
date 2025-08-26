import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bot import Config, TraderBot, SymbolFetcher


def test_risk_pct_affects_trade_size(monkeypatch):
    symbol = "TEST-USD"
    price = 10.0
    timestamp = pd.Timestamp("2024-01-01")

    # prevent background thread/network activity
    monkeypatch.setattr(SymbolFetcher, "start", lambda self: None)

    config_low = Config(
        symbol=symbol,
        risk_pct=0.01,
        atr_multiplier=0,
        max_exposure=1.0,
        stop_loss_pct=0.05,
    )
    bot_low = TraderBot(config_low)
    bot_low.execute_trade("buy", price, timestamp, symbol)
    amount_low = bot_low.account.positions[symbol]["amount"]

    config_high = Config(
        symbol=symbol,
        risk_pct=0.02,
        atr_multiplier=0,
        max_exposure=1.0,
        stop_loss_pct=0.05,
    )
    bot_high = TraderBot(config_high)
    bot_high.execute_trade("buy", price, timestamp, symbol)
    amount_high = bot_high.account.positions[symbol]["amount"]

    assert amount_high == pytest.approx(amount_low * 2)

