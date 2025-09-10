"""Bollinger Band breakout strategy."""
from typing import Optional
import pandas as pd


def generate_signal(df: pd.DataFrame, config) -> Optional[str]:
    """Return "buy" if price closes above upper band and "sell" if below lower band."""
    period = getattr(config, "bb_period", 20)
    std_mult = getattr(config, "bb_std_multiplier", 2.0)

    rolling = df["close"].rolling(period)
    mid = rolling.mean()
    std = rolling.std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std

    if len(df) <= period:
        return None

    close_now = df["close"].iloc[-1]
    upper_prev = upper.iloc[-2]
    lower_prev = lower.iloc[-2]

    if pd.notna(upper_prev) and close_now > upper_prev:
        return "buy"
    if pd.notna(lower_prev) and close_now < lower_prev:
        return "sell"
    return None
