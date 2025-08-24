import csv
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bot import Config, PaperAccount


def test_sell_nonexistent_symbol(tmp_path):
    config = Config()
    account = PaperAccount(balance=1000.0, max_exposure=1.0, config=config)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        ts = pd.Timestamp("2024-01-01")
        assert account.buy(price=100.0, amount=1.0, timestamp=ts, symbol="BTC-USD")

        result = account.sell(
            price=110.0,
            timestamp=ts + pd.Timedelta(minutes=1),
            symbol="ETH-USD",
        )
        assert result is False

        assert len(account.log) == 1
        with open("trade_log.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["side"] == "buy"
    finally:
        os.chdir(old_cwd)
