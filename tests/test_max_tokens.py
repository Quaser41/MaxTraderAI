import csv
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bot import Config, TraderBot, SymbolFetcher


def test_execute_trade_respects_max_tokens(tmp_path, monkeypatch):
    symbol = "TEST-USD"
    price = 10.0
    risk_pct = 0.015
    max_tokens = 5.0  # risk-based amount will exceed this and should cap

    # prevent background thread/network activity
    monkeypatch.setattr(SymbolFetcher, "start", lambda self: None)

    config = Config(symbol=symbol, risk_pct=risk_pct, max_tokens=max_tokens, atr_multiplier=0)
    bot = TraderBot(config)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="min"),
            "open": [price, price],
            "high": [price + 1, price + 2],
            "low": [price - 1, price - 2],
            "close": [price, price],
            "volume": [0, 0],
        }
    )
    monkeypatch.setattr(bot, "fetch_candles", lambda symbol=None: df)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        timestamp = pd.Timestamp("2024-01-01")
        bot.execute_trade("buy", price, timestamp, symbol)
        with open(os.path.join("logs", "trade_log.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
    finally:
        os.chdir(old_cwd)

    assert rows, "No trades logged"
    assert float(rows[-1]["amount"]) == max_tokens
