"""EMA crossover strategy filtered by RSI."""
from typing import Optional
import logging
import pandas as pd


def generate_signal(df: pd.DataFrame, config) -> Optional[str]:
    """Generate a moving average crossover signal filtered by RSI."""
    df["ema_fast"] = df["close"].ewm(span=config.ema_fast_span).mean()
    df["ema_slow"] = df["close"].ewm(span=config.ema_slow_span).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=config.rsi_period).mean()
    avg_loss = loss.rolling(window=config.rsi_period).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    rsi_series = df["rsi"]
    rsi_mean = rsi_series.rolling(config.rsi_period).mean().iloc[-1]
    rsi_std = rsi_series.rolling(config.rsi_period).std().iloc[-1]
    if pd.isna(rsi_mean) or pd.isna(rsi_std):
        buy_thresh = config.rsi_buy_threshold
        sell_thresh = config.rsi_sell_threshold
    else:
        buy_thresh = max(
            config.rsi_buy_threshold,
            rsi_mean + config.rsi_std_multiplier * rsi_std,
        )
        sell_thresh = min(
            config.rsi_sell_threshold,
            rsi_mean - config.rsi_std_multiplier * rsi_std,
        )

    ema_diff = df["ema_fast"] - df["ema_slow"]
    vol = df["close"].pct_change().rolling(config.ema_slow_span).std().iloc[-1]
    ema_threshold = vol * config.ema_threshold_mult if pd.notna(vol) else 0.0
    ema_curr = ema_diff.iloc[-1]
    ema_prev = ema_diff.iloc[-2]
    rsi_now = df["rsi"].iloc[-1]
    buy_ema = ema_curr > ema_threshold and ema_prev <= ema_threshold
    buy_rsi = rsi_now > buy_thresh
    if buy_ema and (not config.use_rsi_filter or buy_rsi):
        return "buy"
    if not buy_ema:
        logging.debug(
            "Buy EMA condition failed: diff %.4f (prev %.4f) threshold %.4f",
            ema_curr,
            ema_prev,
            ema_threshold,
        )
    if config.use_rsi_filter and not buy_rsi:
        logging.debug(
            "Buy RSI condition failed: %.2f <= %.2f",
            rsi_now,
            buy_thresh,
        )
    sell_ema = ema_curr < -ema_threshold and ema_prev >= -ema_threshold
    sell_rsi = rsi_now < sell_thresh
    if sell_ema and (not config.use_rsi_filter or sell_rsi):
        return "sell"
    if not sell_ema:
        logging.debug(
            "Sell EMA condition failed: diff %.4f (prev %.4f) threshold -%.4f",
            ema_curr,
            ema_prev,
            ema_threshold,
        )
    if config.use_rsi_filter and not sell_rsi:
        logging.debug(
            "Sell RSI condition failed: %.2f >= %.2f",
            rsi_now,
            sell_thresh,
        )
    return None
