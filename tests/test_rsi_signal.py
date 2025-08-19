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


def test_rsi_threshold(monkeypatch):
    # prevent background thread/network activity
    monkeypatch.setattr(SymbolFetcher, "start", lambda self: None)
    monkeypatch.setattr(SymbolFetcher, "wait_until_ready", lambda self, timeout=None: None)

    df_buy = make_df([100] * 50 + [110])
    df_sell = make_df([100] * 50 + [90])

    bot_block_buy = TraderBot(Config(symbol="T-USD", rsi_buy_threshold=100))
    assert bot_block_buy.generate_signal(df_buy) is None

    bot_allow_buy = TraderBot(Config(symbol="T-USD", rsi_buy_threshold=50))
    assert bot_allow_buy.generate_signal(df_buy) == "buy"

    bot_block_sell = TraderBot(Config(symbol="T-USD", rsi_sell_threshold=-1))
    assert bot_block_sell.generate_signal(df_sell) is None

    bot_allow_sell = TraderBot(Config(symbol="T-USD", rsi_sell_threshold=50))
    assert bot_allow_sell.generate_signal(df_sell) == "sell"
