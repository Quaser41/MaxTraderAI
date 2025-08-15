import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import bot
from bot import Config, TraderBot, SymbolFetcher


def test_max_holding_period_auto_exit(monkeypatch):
    symbol = "TEST-USD"

    # prevent background thread/network activity
    monkeypatch.setattr(SymbolFetcher, "start", lambda self: None)

    config = Config(symbol=symbol, max_holding_minutes=1)
    bot_instance = TraderBot(config)

    entry_time = pd.Timestamp("2024-01-01 00:00:00")
    assert bot_instance.account.buy(
        price=100.0, amount=1.0, timestamp=entry_time, symbol=symbol
    )

    later_time = entry_time + pd.Timedelta(minutes=2)
    df = pd.DataFrame(
        {
            "timestamp": [later_time],
            "open": [100.0],
            "high": [100.0],
            "low": [100.0],
            "close": [100.0],
            "volume": [0.0],
        }
    )

    monkeypatch.setattr(TraderBot, "fetch_candles", lambda self, symbol=None: df)

    # stop the run loop after one iteration
    def stop_sleep(_seconds):
        raise StopIteration

    monkeypatch.setattr(bot.time, "sleep", stop_sleep)

    with pytest.raises(StopIteration):
        bot_instance.run()

    assert symbol not in bot_instance.account.positions
