import os
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add src directory to path for importing bot module
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bot import Config, PaperAccount


def test_log_entry_requires_fee(tmp_path):
    config = Config()
    account = PaperAccount(balance=1000.0, max_exposure=1.0, config=config)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        entry = {
            "timestamp": pd.Timestamp("2024-01-01").isoformat(),
            "symbol": "TEST-USD",
            "side": "buy",
            "price": 100.0,
            "amount": 1.0,
            "profit": "",
            "fee": "",
            "duration": "",
        }
        with pytest.raises(ValueError):
            account._log_to_file(entry)
        assert not os.path.exists("trade_log.csv")
    finally:
        os.chdir(old_cwd)

