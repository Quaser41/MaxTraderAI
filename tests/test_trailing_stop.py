import sys
from pathlib import Path

import pandas as pd

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bot import Config, PaperAccount


def test_trailing_stop_overrides_price():
    config = Config(symbol="TEST-USD")
    account = PaperAccount(balance=1000, max_exposure=1.0, config=config)
    ts = pd.Timestamp("2024-01-01T00:00:00Z")
    account.buy(price=100, amount=1, timestamp=ts, symbol="TEST-USD")
    sell_ts = ts + pd.Timedelta(minutes=1)
    account.sell(price=95, timestamp=sell_ts, symbol="TEST-USD", trailing_stop=90)
    assert account.log[-1]["price"] == 90
