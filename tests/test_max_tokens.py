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
    stake_usd = 100.0
    max_tokens = 5.0  # stake_usd/price = 10, so should cap at 5

    # prevent background thread/network activity
    monkeypatch.setattr(SymbolFetcher, "start", lambda self: None)

    config = Config(symbol=symbol, stake_usd=stake_usd, max_tokens=max_tokens)
    bot = TraderBot(config)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        timestamp = pd.Timestamp("2024-01-01")
        bot.execute_trade("buy", price, timestamp, symbol)
        with open("trade_log.csv", newline="") as f:
            rows = list(csv.DictReader(f))
    finally:
        os.chdir(old_cwd)

    assert rows, "No trades logged"
    assert float(rows[-1]["amount"]) == max_tokens
