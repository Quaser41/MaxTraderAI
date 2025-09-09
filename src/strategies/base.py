from typing import Optional
import pandas as pd


def generate_signal(df: pd.DataFrame, config) -> Optional[str]:
    """Return a trading signal based on ``df`` and ``config``.

    Strategy modules should implement this function and return "buy",
    "sell" or ``None``.
    """
    raise NotImplementedError
