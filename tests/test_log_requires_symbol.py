import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bot import Config, PaperAccount


def test_log_entry_requires_symbol(tmp_path):
    config = Config()
    account = PaperAccount(balance=1000.0, max_exposure=1.0, config=config)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        entry = {
            "timestamp": pd.Timestamp("2024-01-01").isoformat(),
            "symbol": "",
            "side": "buy",
            "price": 100.0,
            "amount": 1.0,
            "profit": "",
            "duration": "",
        }
        with pytest.raises(ValueError):
            account._log_to_file(entry)
        assert not os.path.exists(os.path.join("logs", "trade_log.csv"))
        with pytest.raises(ValueError):
            account.buy(price=100.0, amount=1.0, timestamp=pd.Timestamp("2024-01-01"), symbol="")
        assert account.log == []
    finally:
        os.chdir(old_cwd)
