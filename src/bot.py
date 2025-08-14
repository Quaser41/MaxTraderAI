"""

Minimal autonomous crypto trading bot using Yahoo Finance data.


This code is for educational purposes only and does not constitute financial advice.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict
import time
import csv
import os
import logging
import pandas as pd
import yfinance as yf
import ccxt

logging.basicConfig(level=logging.INFO)


@dataclass
class Config:

    symbol: str = "BTC-USD"
    timeframe: str = "1h"
    stake: float = 0.001  # trade size in asset units
    starting_balance: float = 1000.0
    max_exposure: float = 0.75  # fraction of account allowed in a single trade
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit target
    max_drawdown_pct: float = 0.2  # stop trading if drawdown exceeds 20%


class PaperAccount:
    """Simple paper trading ledger."""

    def __init__(self, balance: float, max_exposure: float) -> None:
        self.initial_balance = balance
        self.balance = balance
        self.max_exposure = max_exposure
        self.position: Optional[Dict] = None
        self.log: List[Dict] = []
        self.peak_balance = balance

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

    def buy(
        self,
        price: float,
        amount: float,
        timestamp: pd.Timestamp,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> bool:
        cost = price * amount
        if cost > self.initial_balance * self.max_exposure or cost > self.balance:
            print("Buy skipped: exposure limit or insufficient balance")
            return False
        self.balance -= cost
        self.position = {
            "price": price,
            "amount": amount,
            "timestamp": timestamp,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }
        self.peak_balance = max(self.peak_balance, self.balance)
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
        self.peak_balance = max(self.peak_balance, self.balance)
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
        self.print_performance()
        return True

    def current_drawdown(self) -> float:
        return (self.peak_balance - self.balance) / self.peak_balance

    def print_performance(self) -> None:
        sells = [t for t in self.log if t["side"] == "sell"]
        if not sells:
            return
        net_profit = self.balance - self.initial_balance
        wins = [t for t in sells if t["profit"] > 0]
        win_rate = len(wins) / len(sells) * 100
        print(
            f"Trades: {len(sells)} | Net Profit: {net_profit:.2f} | Win rate: {win_rate:.1f}%"
        )


class TraderBot:
    def __init__(self, config: Config):
        self.config = config
        self.account = PaperAccount(config.starting_balance, config.max_exposure)

    def fetch_candles_ccxt(self) -> pd.DataFrame:
        """Fetch recent OHLCV data from Binance via CCXT."""
        exchange = ccxt.binance()
        symbol = self.config.symbol.replace("-", "/")
        if symbol.endswith("USD"):
            symbol = symbol[:-3] + "/USDT"
        data = exchange.fetch_ohlcv(
            symbol=symbol, timeframe=self.config.timeframe, limit=100
        )
        df = pd.DataFrame(
            data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def fetch_candles(self) -> pd.DataFrame:
        """Fetch recent OHLCV data from Yahoo Finance with CCXT fallback.

        Data is downloaded with ``auto_adjust`` set to ``False`` to preserve the
        raw price data returned by Yahoo Finance. Set this argument to ``True``
        if adjusted prices (accounting for splits/dividends) are desired in the
        future. If Yahoo Finance fails, data is fetched from Binance via CCXT.
        """
        df = pd.DataFrame()
        for attempt in range(3):
            try:
                df = yf.download(
                    tickers=self.config.symbol,
                    period="7d",
                    interval=self.config.timeframe,
                    progress=False,
                    auto_adjust=False,  # preserve raw prices for trading
                    timeout=10,
                )
                if not df.empty:
                    break
            except Exception as exc:
                logging.error(
                    "Data fetch failed on attempt %s: %s", attempt + 1, exc
                )
                time.sleep(1)
        if df.empty:
            logging.info("Falling back to CCXT for candle data")
            try:
                df = self.fetch_candles_ccxt()
            except Exception as exc:
                logging.error("CCXT data fetch failed: %s", exc)
                return pd.DataFrame()
        else:
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )
            df = df.reset_index().rename(
                columns={"Datetime": "timestamp", "Date": "timestamp"}
            )

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
            stop = price * (1 - self.config.stop_loss_pct)
            target = price * (1 + self.config.take_profit_pct)
            self.account.buy(
                price, self.config.stake, timestamp, stop_loss=stop, take_profit=target
            )
        elif side == "sell":
            self.account.sell(price, self.config.stake, timestamp)

    def run(self) -> None:
        """Run the trading loop."""
        while True:
            df = self.fetch_candles()
            if df.empty:
                time.sleep(60)
                continue
            price = df["close"].iloc[-1]
            timestamp = df["timestamp"].iloc[-1]
            if self.account.position:
                pos = self.account.position
                if (
                    pos.get("stop_loss") is not None and price <= pos["stop_loss"]
                ) or (
                    pos.get("take_profit") is not None and price >= pos["take_profit"]
                ):
                    self.account.sell(price, pos["amount"], timestamp)
                    continue
            signal = self.generate_signal(df)
            if signal:
                self.execute_trade(signal, price, timestamp)
            if self.account.current_drawdown() > self.config.max_drawdown_pct:
                print("Max drawdown exceeded. Stopping bot.")
                break
            time.sleep(60)


if __name__ == "__main__":

    config = Config()

    bot = TraderBot(config)
    bot.run()
