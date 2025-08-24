import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bot import Config, TraderBot, SymbolFetcher

def make_df():
    timestamps = pd.date_range("2024-01-01", periods=3, freq="min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": [10, 11, 12],
            "high": [11, 13, 14],
            "low": [9, 10, 11],
            "close": [10, 12, 13],
            "volume": [0, 0, 0],
        }
    )

def compute_atr(df):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=14, min_periods=1).mean().iloc[-1]

def test_execute_trade_uses_atr(monkeypatch):
    monkeypatch.setattr(SymbolFetcher, "start", lambda self: None)
    df = make_df()
    atr = compute_atr(df)
    symbol = "TEST-USD"
    price = df["close"].iloc[-1]
    ts = df["timestamp"].iloc[-1]
    atr_mult = 2.0
    config = Config(symbol=symbol, atr_multiplier=atr_mult)
    bot = TraderBot(config)
    monkeypatch.setattr(bot, "fetch_candles", lambda symbol=None: df)
    bot.execute_trade("buy", price, ts, symbol)
    pos = bot.account.positions[symbol]
    assert pos["stop_loss"] == pytest.approx(price - atr * atr_mult)
    assert pos["take_profit"] == pytest.approx(price + atr * atr_mult)
