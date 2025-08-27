import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bot import Config, TraderBot, SymbolFetcher


def make_df(closes):
    timestamps = pd.date_range("2024-01-01", periods=len(closes), freq="min")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [0] * len(closes),
        }
    )


def test_ema_crossover(monkeypatch):
    # prevent background thread/network activity
    monkeypatch.setattr(SymbolFetcher, "start", lambda self: None)
    monkeypatch.setattr(SymbolFetcher, "wait_until_ready", lambda self, timeout=None: None)

    df_buy = make_df([100] * 50 + [110])
    df_sell = make_df([100] * 50 + [90])

    bot = TraderBot(Config(symbol="T-USD"))
    assert bot.generate_signal(df_buy) == "buy"
    assert bot.generate_signal(df_sell) == "sell"
