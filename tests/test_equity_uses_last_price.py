import sys
from pathlib import Path

import pandas as pd
import pytest

# add src directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bot import Config, PaperAccount


def test_equity_reflects_last_price():
    symbol = "TEST-USD"
    config = Config()
    account = PaperAccount(balance=1000.0, max_exposure=1.0, config=config)
    timestamp = pd.Timestamp("2024-01-01")
    entry_price = 100.0
    amount = 1.0

    assert account.buy(price=entry_price, amount=amount, timestamp=timestamp, symbol=symbol)
    # Equity initially reflects entry price
    assert account.get_equity() == pytest.approx(1000.0)

    # Update last_price and ensure equity uses it
    account.positions[symbol]["last_price"] = 110.0
    expected_equity = account.balance + 110.0 * amount
    assert account.get_equity() == pytest.approx(expected_equity)
