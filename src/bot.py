"""

Minimal autonomous crypto trading bot using Yahoo Finance data.


This code is for educational purposes only and does not constitute financial advice.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict
import time
import csv
import os


import pandas as pd
import yfinance as yf


@dataclass
class Config:

    symbol: str = "BTC-USD"
    timeframe: str = "1h"
    stake: float = 0.001  # trade size in asset units
    starting_balance: float = 1000.0
    max_exposure: float = 0.75  # fraction of account allowed in a single trade


class PaperAccount:
    """Simple paper trading ledger."""

    def __init__(self, balance: float, max_exposure: float) -> None:
        self.initial_balance = balance
        self.balance = balance
        self.max_exposure = max_exposure
        self.position: Optional[Dict] = None
        self.log: List[Dict] = []

    def _log_to_file(self, entry: Dict) -> None:
        path = "trade_log.csv"
        file_exists = os.path.isfile(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["timestamp", "side", "price", "amount", "profit", "duration"],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(entry)

    def buy(self, price: float, amount: float, timestamp: pd.Timestamp) -> bool:
        cost = price * amount
        if cost > self.initial_balance * self.max_exposure or cost > self.balance:
            print("Buy skipped: exposure limit or insufficient balance")
            return False
        self.balance -= cost
        self.position = {"price": price, "amount": amount, "timestamp": timestamp}
        entry = {
            "timestamp": timestamp.isoformat(),
            "side": "buy",
            "price": price,
            "amount": amount,
            "profit": "",
            "duration": "",
        }
        self.log.append(entry)
        self._log_to_file(entry)
        print(f"BUY {amount} at {price:.2f} -- balance {self.balance:.2f}")
        return True

    def sell(self, price: float, amount: float, timestamp: pd.Timestamp) -> bool:
        if not self.position:
            print("Sell skipped: no open position")
            return False
        entry_price = self.position["price"]
        profit = (price - entry_price) * amount
        self.balance += price * amount
        duration = timestamp - self.position["timestamp"]
        entry = {
            "timestamp": timestamp.isoformat(),
            "side": "sell",
            "price": price,
            "amount": amount,
            "profit": profit,
            "duration": duration.total_seconds(),
        }
        self.log.append(entry)
        self._log_to_file(entry)
        print(
            f"SELL {amount} at {price:.2f} -- PnL {profit:.2f} -- balance {self.balance:.2f}"
        )
        self.position = None
        return True


class TraderBot:
    def __init__(self, config: Config):
        self.config = config
        self.account = PaperAccount(config.starting_balance, config.max_exposure)

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

    def execute_trade(self, side: str, price: float, timestamp: pd.Timestamp) -> None:
        """Execute a paper trade through the PaperAccount."""
        if side == "buy":
            self.account.buy(price, self.config.stake, timestamp)
        elif side == "sell":
            self.account.sell(price, self.config.stake, timestamp)

    def run(self) -> None:
        """Run the trading loop."""
        while True:
            df = self.fetch_candles()
            signal = self.generate_signal(df)
            price = df["close"].iloc[-1]
            timestamp = df["timestamp"].iloc[-1]
            if signal:
                self.execute_trade(signal, price, timestamp)
            time.sleep(60)


if __name__ == "__main__":

    config = Config()

    bot = TraderBot(config)
    bot.run()
