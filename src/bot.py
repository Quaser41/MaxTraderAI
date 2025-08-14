"""

Minimal autonomous crypto trading bot using CCXT data.


This code is for educational purposes only and does not constitute financial advice.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict
import time
import csv
import os
import logging
import threading
import pandas as pd
import requests
import ccxt

logging.basicConfig(level=logging.INFO)


@dataclass
class Config:

    symbol: str = "BTC-USD"
    timeframe: str = "1h"
    exchange: str = "binanceus"
    stake: float = 0.001  # trade size in asset units
    starting_balance: float = 1000.0
    max_exposure: float = 0.75  # fraction of account allowed in a single trade
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit target
    max_drawdown_pct: float = 0.2  # stop trading if drawdown exceeds 20%



class SymbolFetcher:
    """Background thread that refreshes top-volume symbols from BinanceUS."""

    def __init__(self, refresh: int = 3600, limit: int = 10) -> None:
        self.refresh = refresh
        self.limit = limit
        self.symbols: List[str] = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                resp = requests.get(
                    "https://api.binance.us/api/v3/ticker/24hr", timeout=10
                )
                data = resp.json()
                usdt_pairs = [d for d in data if d.get("symbol", "").endswith("USDT")]
                usdt_pairs.sort(
                    key=lambda d: float(d.get("quoteVolume", 0)), reverse=True
                )
                exchange = ccxt.binanceus()
                exchange.load_markets()
                validated: List[str] = []
                for d in usdt_pairs:
                    base = d["symbol"][:-4]
                    symbol = base + "-USD"
                    try:
                        exchange.fetch_ticker(base + "/USDT")
                    except Exception as exc:
                        logging.info("Skipping symbol %s: %s", symbol, exc)
                        continue
                    validated.append(symbol)
                    if len(validated) >= self.limit:
                        break
                self.symbols = validated
                if self.symbols:
                    logging.info("Fetched symbols: %s", ", ".join(self.symbols))
            except Exception as exc:
                logging.error("Symbol fetch failed: %s", exc)
            time.sleep(self.refresh)


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
        self.symbol_fetcher = SymbolFetcher()
        self.symbol_fetcher.start()

    def fetch_candles_ccxt(self, exchange_name: str = "binanceus") -> pd.DataFrame:
        """Fetch recent OHLCV data from an exchange via CCXT.

        Parameters
        ----------
        exchange_name: str, optional
            Name of the exchange from the CCXT library. Defaults to ``"binanceus"``.
        """
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class()
        symbol = self.config.symbol.replace("-", "/")
        if symbol.endswith("USD"):
            symbol = symbol[:-3] + "USDT"
        data = exchange.fetch_ohlcv(
            symbol=symbol, timeframe=self.config.timeframe, limit=100
        )
        df = pd.DataFrame(
            data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def fetch_candles(self) -> pd.DataFrame:
        """Fetch recent OHLCV data from the configured exchange via CCXT."""
        try:
            return self.fetch_candles_ccxt(self.config.exchange)
        except Exception as exc:
            logging.error("CCXT data fetch failed: %s", exc)
            return pd.DataFrame()

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
            symbols = self.symbol_fetcher.symbols or [self.config.symbol]
            for symbol in symbols:
                self.config.symbol = symbol
                df = self.fetch_candles()
                if df.empty:
                    time.sleep(1)
                    continue
                price = df["close"].iloc[-1]
                timestamp = df["timestamp"].iloc[-1]
                if self.account.position:
                    pos = self.account.position
                    if (
                        pos.get("stop_loss") is not None and price <= pos["stop_loss"]
                    ) or (
                        pos.get("take_profit") is not None
                        and price >= pos["take_profit"]
                    ):
                        self.account.sell(price, pos["amount"], timestamp)
                        continue
                signal = self.generate_signal(df)
                if signal:
                    self.execute_trade(signal, price, timestamp)
                if self.account.current_drawdown() > self.config.max_drawdown_pct:
                    print("Max drawdown exceeded. Stopping bot.")
                    return
                time.sleep(1)
            time.sleep(60)


if __name__ == "__main__":

    config = Config()

    bot = TraderBot(config)
    bot.run()
