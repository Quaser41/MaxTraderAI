"""Simple RSI mean-reversion strategy."""
from typing import Optional
import pandas as pd


def generate_signal(df: pd.DataFrame, config) -> Optional[str]:
    """Buy when RSI < 30, sell when RSI > 70."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=config.rsi_period).mean()
    avg_loss = loss.rolling(window=config.rsi_period).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    rsi_now = df["rsi"].iloc[-1]
    if rsi_now < 30:
        return "buy"
    if rsi_now > 70:
        return "sell"
    return None
