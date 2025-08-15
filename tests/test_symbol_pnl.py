import sys
from pathlib import Path
import pandas as pd

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bot import Config, PaperAccount


def test_pnl_by_symbol_window():
    config = Config()
    account = PaperAccount(balance=1000.0, max_exposure=1.0, config=config)
    ts = pd.Timestamp("2024-01-01")

    # first trade: profit 10
    account.buy(price=100.0, amount=1.0, timestamp=ts, symbol="AAA-USD")
    account.sell(price=110.0, timestamp=ts, symbol="AAA-USD")

    # second trade: loss 10
    account.buy(price=100.0, amount=1.0, timestamp=ts, symbol="AAA-USD")
    account.sell(price=90.0, timestamp=ts, symbol="AAA-USD")

    pnl_all = account.pnl_by_symbol().get("AAA-USD")
    pnl_last = account.pnl_by_symbol(window=1).get("AAA-USD")

    assert pnl_all == 0.0
    assert pnl_last == -10.0
