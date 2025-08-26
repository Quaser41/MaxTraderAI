import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src directory to path for importing bot module
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bot import Config, PaperAccount


@pytest.mark.parametrize("spread_pct", [0.0, 0.02])
def test_sell_uses_trailing_stop_price(tmp_path, spread_pct):
    config = Config(spread_pct=spread_pct)
    account = PaperAccount(balance=1000.0, max_exposure=1.0, config=config)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ts = pd.Timestamp("2024-01-01")
        symbol = "BTC-USD"
        assert account.buy(price=100.0, amount=1.0, timestamp=ts, symbol=symbol)

        trailing_stop = 90.0
        assert account.sell(
            price=120.0,
            timestamp=ts + pd.Timedelta(minutes=1),
            symbol=symbol,
            fee_pct=0.0,
            trailing_stop=trailing_stop,
        )

        sell_entry = account.log[-1]
        expected_price = trailing_stop * (1 - spread_pct / 2)
        assert sell_entry["price"] == pytest.approx(expected_price)
    finally:
        os.chdir(old_cwd)


@pytest.mark.parametrize("spread_pct", [0.0, 0.02])
def test_trailing_stop_equals_price(tmp_path, spread_pct):
    config = Config(spread_pct=spread_pct)
    account = PaperAccount(balance=1000.0, max_exposure=1.0, config=config)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ts = pd.Timestamp("2024-01-01")
        symbol = "BTC-USD"
        assert account.buy(price=100.0, amount=1.0, timestamp=ts, symbol=symbol)

        price = 90.0
        trailing_stop = price
        assert account.sell(
            price=price,
            timestamp=ts + pd.Timedelta(minutes=1),
            symbol=symbol,
            fee_pct=0.0,
            trailing_stop=trailing_stop,
        )

        sell_entry = account.log[-1]
        expected_price = trailing_stop * (1 - spread_pct / 2)
        assert sell_entry["price"] == pytest.approx(expected_price)
    finally:
        os.chdir(old_cwd)
