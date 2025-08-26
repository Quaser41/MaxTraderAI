import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bot import Config, PaperAccount


def test_existing_trade_log_archived(tmp_path):
    # create an old log file
    old_log = tmp_path / "trade_log.csv"
    old_log.write_text("old data")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        PaperAccount(balance=1000.0, max_exposure=1.0, config=Config())
        assert not old_log.exists()
        archived = list(tmp_path.glob("trade_log_*.csv"))
        assert len(archived) == 1
    finally:
        os.chdir(old_cwd)

