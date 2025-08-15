import csv
import os
import sys
from pathlib import Path

import pandas as pd

# Add src directory to path for importing bot module
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bot import Config, PaperAccount


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
