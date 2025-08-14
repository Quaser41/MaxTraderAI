"""
Minimal autonomous crypto trading bot using Yahoo Finance data.

This code is for educational purposes only and does not constitute financial advice.
"""
from dataclasses import dataclass
from typing import Optional
import time

import pandas as pd
import yfinance as yf


@dataclass
class Config:
    symbol: str = "BTC-USD"
    timeframe: str = "1h"
    stake: float = 0.001  # size of simulated trade


class TraderBot:
    def __init__(self, config: Config):
        self.config = config

    def fetch_candles(self) -> pd.DataFrame:
        """Fetch recent OHLCV data from Yahoo Finance."""
        df = yf.download(
            tickers=self.config.symbol,
            period="7d",
            interval=self.config.timeframe,
            progress=False,
        )
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df = df.reset_index().rename(columns={"Datetime": "timestamp", "Date": "timestamp"})
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
        """Simulate a market order."""
        print(f"{side.upper()} {self.config.stake} {self.config.symbol}")

    def run(self) -> None:
        """Run the trading loop."""
        while True:
            df = self.fetch_candles()
            signal = self.generate_signal(df)
            if signal:
                self.execute_trade(signal)
            time.sleep(60)


if __name__ == "__main__":
    config = Config()
    bot = TraderBot(config)
    bot.run()
