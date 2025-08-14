"""
Minimal autonomous crypto trading bot.

This code is for educational purposes only and does not constitute financial advice.
"""
from dataclasses import dataclass
from typing import Optional
import time

import ccxt
import pandas as pd


@dataclass
class Config:
    api_key: str
    secret: str
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    stake: float = 0.001
    stop_loss: float = 0.02
    take_profit: float = 0.04


class TraderBot:
    def __init__(self, config: Config):
        self.exchange = ccxt.binance({
            "apiKey": config.api_key,
            "secret": config.secret,
            "enableRateLimit": True,
        })
        self.config = config

    def fetch_candles(self) -> pd.DataFrame:
        """Fetch recent OHLCV data."""
        ohlcv = self.exchange.fetch_ohlcv(
            self.config.symbol,
            timeframe=self.config.timeframe,
            limit=100,
        )
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def generate_signal(self, df: pd.DataFrame) -> Optional[str]:
        """Generate a simple moving average crossover signal."""
        df["ema_fast"] = df["close"].ewm(span=12).mean()
        df["ema_slow"] = df["close"].ewm(span=26).mean()
        if (
            df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1]
            and df["ema_fast"].iloc[-2] <= df["ema_slow"].iloc[-2]
        ):
            return "buy"
        if (
            df["ema_fast"].iloc[-1] < df["ema_slow"].iloc[-1]
            and df["ema_fast"].iloc[-2] >= df["ema_slow"].iloc[-2]
        ):
            return "sell"
        return None

    def execute_trade(self, side: str) -> None:
        """Execute a market order on the configured exchange."""
        amount = self.config.stake
        if side == "buy":
            self.exchange.create_market_buy_order(self.config.symbol, amount)
        elif side == "sell":
            self.exchange.create_market_sell_order(self.config.symbol, amount)

    def run(self) -> None:
        """Run the trading loop."""
        while True:
            df = self.fetch_candles()
            signal = self.generate_signal(df)
            if signal:
                self.execute_trade(signal)
            time.sleep(60)


if __name__ == "__main__":
    # Example usage - replace with real keys!
    config = Config(api_key="YOUR_API_KEY", secret="YOUR_SECRET")
    bot = TraderBot(config)
    bot.run()
