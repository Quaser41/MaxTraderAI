import sys
from pathlib import Path

import pandas as pd
import pytest

# add src directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bot import Config, PaperAccount


def test_exposure_respects_current_equity_gains_and_drawdowns():
    symbol1 = "AAA-USD"
    symbol2 = "BBB-USD"
    config = Config()
    timestamp = pd.Timestamp("2024-01-01")

    # ---- Gains scenario ----
    account = PaperAccount(balance=1000.0, max_exposure=0.5, config=config)
    assert account.buy(price=100.0, amount=1.0, timestamp=timestamp, symbol=symbol1)

    # increase unrealized profit to boost equity
    account.positions[symbol1]["last_price"] = 200.0  # equity becomes 1100

    # cost 525 is > initial_balance * max_exposure (500) but <= equity * max_exposure (550)
    assert account.buy(
        price=100.0,
        amount=5.25,
        timestamp=timestamp,
        symbol=symbol2,
    )

    # ---- Drawdown scenario ----
    account = PaperAccount(balance=1000.0, max_exposure=0.5, config=config)
    assert account.buy(price=100.0, amount=1.0, timestamp=timestamp, symbol=symbol1)

    # decrease position value to reduce equity
    account.positions[symbol1]["last_price"] = 50.0  # equity becomes 950

    # cost 490 is < initial_balance * max_exposure (500) but > equity * max_exposure (475)
    assert not account.buy(
        price=100.0,
        amount=4.9,
        timestamp=timestamp,
        symbol=symbol2,
    )
    assert symbol2 not in account.positions
