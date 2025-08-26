import csv
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src directory to path for importing bot module
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bot import Config, PaperAccount, TraderBot, SymbolFetcher


def test_trade_log_contains_symbol(tmp_path):
    symbol = "TEST-USD"
    config = Config()
    account = PaperAccount(balance=1000.0, max_exposure=1.0, config=config)

    # operate within temporary directory
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        timestamp = pd.Timestamp("2024-01-01")
        assert account.buy(
            price=100.0, amount=1.0, timestamp=timestamp, symbol=symbol
        )

        with open("trade_log.csv", newline="") as f:
            rows = list(csv.DictReader(f))
    finally:
        os.chdir(old_cwd)

    assert rows and rows[0]["symbol"] == symbol
    assert "fee" in rows[0]


def test_execute_trade_logs_amount(tmp_path, monkeypatch):
    """Ensure TraderBot passes the calculated amount through to the CSV."""
    stake_usd = 50.0
    price = 10.0
    expected_amount = stake_usd / price
    symbol = "TEST-USD"
    fee_pct = 0.001

    # prevent background thread/network activity
    monkeypatch.setattr(SymbolFetcher, "start", lambda self: None)

    config = Config(stake_usd=stake_usd, symbol=symbol, fee_pct=fee_pct)
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
        with open("trade_log.csv", newline="") as f:
            rows = list(csv.DictReader(f))
    finally:
        os.chdir(old_cwd)

    assert rows, "No trades logged"
    assert float(rows[-1]["amount"]) == expected_amount
    expected_fee = stake_usd * fee_pct
    assert float(rows[-1]["fee"]) == pytest.approx(expected_fee)


def test_buy_logs_fee(tmp_path):
    symbol = "TEST-USD"
    config = Config()
    account = PaperAccount(balance=1000.0, max_exposure=1.0, config=config)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        buy_time = pd.Timestamp("2024-01-01")
        price = 100.0
        amount = 1.0
        fee_pct = 0.001
        assert account.buy(
            price=price,
            amount=amount,
            timestamp=buy_time,
            symbol=symbol,
            fee_pct=fee_pct,
        )
        with open("trade_log.csv", newline="") as f:
            rows = list(csv.DictReader(f))
    finally:
        os.chdir(old_cwd)

    assert len(rows) == 1
    buy_row = rows[0]
    expected_fee = price * amount * fee_pct
    assert float(buy_row["fee"]) == pytest.approx(expected_fee)

def test_sell_logs_fee(tmp_path):
    symbol = "TEST-USD"
    config = Config()
    account = PaperAccount(balance=1000.0, max_exposure=1.0, config=config)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        buy_time = pd.Timestamp("2024-01-01")
        sell_time = buy_time + pd.Timedelta(hours=1)
        assert account.buy(
            price=100.0, amount=1.0, timestamp=buy_time, symbol=symbol
        )
        sell_price = 110.0
        fee_pct = 0.001
        assert account.sell(
            price=sell_price,
            timestamp=sell_time,
            symbol=symbol,
            fee_pct=fee_pct,
        )
        with open("trade_log.csv", newline="") as f:
            rows = list(csv.DictReader(f))
    finally:
        os.chdir(old_cwd)

    assert len(rows) == 2
    sell_row = rows[1]
    expected_fee = sell_price * 1.0 * fee_pct
    assert "fee" in sell_row
    assert float(sell_row["fee"]) == pytest.approx(expected_fee)


def test_spread_adjusts_pnl(tmp_path):
    symbol = "TEST-USD"
    spread_pct = 0.02
    config = Config(spread_pct=spread_pct)
    account = PaperAccount(balance=1000.0, max_exposure=1.0, config=config)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        buy_time = pd.Timestamp("2024-01-01")
        sell_time = buy_time + pd.Timedelta(hours=1)
        assert account.buy(price=100.0, amount=1.0, timestamp=buy_time, symbol=symbol)
        assert account.sell(price=110.0, timestamp=sell_time, symbol=symbol)
        with open("trade_log.csv", newline="") as f:
            rows = list(csv.DictReader(f))
    finally:
        os.chdir(old_cwd)

    assert len(rows) == 2
    buy_row, sell_row = rows
    expected_buy_price = 100.0 * (1 + spread_pct / 2)
    expected_sell_price = 110.0 * (1 - spread_pct / 2)
    expected_profit = expected_sell_price - expected_buy_price
    assert float(buy_row["price"]) == pytest.approx(expected_buy_price)
    assert float(sell_row["price"]) == pytest.approx(expected_sell_price)
    assert float(sell_row["profit"]) == pytest.approx(expected_profit)
