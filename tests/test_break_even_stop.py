import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import bot
from bot import Config, TraderBot, SymbolFetcher


def test_break_even_moves_stop_loss(monkeypatch):
    symbol = "TEST-USD"
    monkeypatch.setattr(SymbolFetcher, "start", lambda self: None)

    config = Config(symbol=symbol, break_even_pct=0.02)
    bot_instance = TraderBot(config)

    entry_time = pd.Timestamp("2024-01-01 00:00:00")
    assert bot_instance.account.buy(
        price=100.0,
        amount=1.0,
        timestamp=entry_time,
        symbol=symbol,
        stop_loss=95.0,
    )

    pos = bot_instance.account.positions[symbol]
    target_price = pos["price"] * (1 + config.break_even_pct + 0.01)
    df = pd.DataFrame(
        {
            "timestamp": [entry_time + pd.Timedelta(minutes=1)],
            "open": [target_price],
            "high": [target_price],
            "low": [target_price],
            "close": [target_price],
            "volume": [0.0],
        }
    )

    monkeypatch.setattr(TraderBot, "fetch_candles", lambda self, symbol=None: df)
    monkeypatch.setattr(TraderBot, "generate_signal", lambda self, df: None)

    def stop_sleep(_seconds):
        raise StopIteration

    monkeypatch.setattr(bot.time, "sleep", stop_sleep)

    with pytest.raises(StopIteration):
        bot_instance.run()

    pos = bot_instance.account.positions[symbol]
    assert pos["stop_loss"] == pytest.approx(pos["price"])

