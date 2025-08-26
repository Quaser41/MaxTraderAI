import csv
import os
import sys
from pathlib import Path

import pandas as pd

# Add src directory to path for importing bot module
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bot import Config, PaperAccount


def test_log_rotation_creates_new_header(tmp_path):
    old_header = ["timestamp", "symbol", "side", "price", "amount", "profit"]
    old_row = ["2024-01-01T00:00:00", "OLD-USD", "buy", "100", "1", "0"]
    log_path = tmp_path / "trade_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(old_header)
        writer.writerow(old_row)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        config = Config()
        account = PaperAccount(balance=1000.0, max_exposure=1.0, config=config)

        rotated = list((tmp_path / "logs").glob("trade_log_*.csv"))
        assert rotated, "Old log was not rotated"

        timestamp = pd.Timestamp("2024-02-01")
        account.buy(price=100.0, amount=1.0, timestamp=timestamp, symbol="NEW-USD")
        with open(os.path.join("logs", "trade_log.csv"), newline="") as f:
            rows = list(csv.reader(f))
    finally:
        os.chdir(old_cwd)

    new_header = [
        "timestamp",
        "symbol",
        "side",
        "price",
        "amount",
        "profit",
        "fee",
        "duration",
    ]
    assert rows[0] == new_header
    assert len(rows) == 2
    assert len(rows[1]) == len(new_header)
    assert rows[1][1] == "NEW-USD"
