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
    timeframe: str = "1m"
    exchange: str = "binanceus"
    stake_usd: float = 100.0  # trade size in USD
    max_tokens: float = float("inf")  # maximum token quantity per trade
    starting_balance: float = 1000.0
    max_exposure: float = 0.75  # fraction of account allowed in a single trade
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit target
    atr_multiplier: float = 1.0  # ATR multiple for stop-loss and take-profit
    max_drawdown_pct: float = 0.2  # stop trading if drawdown exceeds 20%
    ema_fast_span: int = 12  # fast EMA span for crossover
    ema_slow_span: int = 26  # slow EMA span for crossover
    drawdown_cooldown: int = 300  # seconds to pause after max drawdown
    stop_on_drawdown: bool = True  # stop bot instead of pausing on drawdown
    summary_interval: int = 300  # seconds between status summaries
    pnl_window: int = 10  # number of closed trades to evaluate per-symbol PnL
    min_profit_threshold: float = 0.0  # minimum profit to keep trading a symbol
    fee_pct: float = 0.0  # exchange fee percentage applied on sells
    trailing_stop_pct: float = 0.0  # percentage for trailing stop (0 to disable)
    max_holding_minutes: int = 60  # maximum duration to hold a position
    rsi_period: int = 14  # period for RSI calculation
    rsi_buy_threshold: float = 55.0  # minimum RSI for buy signals
    rsi_sell_threshold: float = 45.0  # maximum RSI for sell signals
    rsi_std_multiplier: float = 1.0  # std-dev multiplier for adaptive RSI
    ema_threshold_mult: float = 0.0  # volatility factor for EMA crossover
    spread_pct: float = 0.0  # estimated bid/ask spread percentage
    min_edge_pct: float = 0.0  # minimum edge required after costs
    min_price: float = 0.0  # minimum token price to include




class SymbolFetcher:
    """Background thread that refreshes top-volume symbols from BinanceUS."""

    def __init__(
        self, refresh: int = 3600, limit: int = 10, min_price: float = 0.0
    ) -> None:
        self.refresh = refresh
        self.limit = limit
        self.min_price = min_price
        self.symbols: List[str] = []
        self._ready = threading.Event()
        self.exchange = ccxt.binanceus()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def wait_until_ready(self, timeout: Optional[float] = None) -> None:
        """Block until at least one symbol has been fetched."""
        if not self._thread.is_alive():
            return
        if not self.symbols:
            logging.info("Waiting for initial symbol data...")
        self._ready.wait(timeout=timeout)
        if self.symbols:
            logging.info(
                "Symbol fetcher initialized with symbols: %s",
                ", ".join(self.symbols),
            )

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
                self.exchange.load_markets()
                validated: List[str] = []
                for d in usdt_pairs:
                    base = d["symbol"][:-4]
                    symbol = base + "-USD"
                    price = float(d.get("lastPrice", 0) or 0)
                    if price < self.min_price:
                        continue
                    try:
                        self.exchange.fetch_ticker(base + "/USDT")
                    except Exception as exc:
                        logging.info("Skipping symbol %s: %s", symbol, exc)
                        continue
                    validated.append(symbol)
                    if len(validated) >= self.limit:
                        break
                self.symbols = validated
                if self.symbols:
                    logging.info("Fetched symbols: %s", ", ".join(self.symbols))
                    if not self._ready.is_set():
                        self._ready.set()
            except Exception as exc:
                logging.error("Symbol fetch failed: %s", exc)
            time.sleep(self.refresh)


class PaperAccount:
    """Simple paper trading ledger."""

    def __init__(self, balance: float, max_exposure: float, config: Config) -> None:
        self.initial_balance = balance
        self.balance = balance
        self.max_exposure = max_exposure
        self.positions: Dict[str, Dict] = {}
        self.log: List[Dict] = []
        self.peak_balance = balance
        # retain reference to the configuration so we can log the active symbol
        self.config = config

        # remove or archive existing trade log to start fresh each run
        log_path = "trade_log.csv"
        if os.path.isfile(log_path):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            archive_path = f"trade_log_{timestamp}.csv"
            os.rename(log_path, archive_path)
            logging.info("Archived existing trade log to %s", archive_path)

    def get_equity(self) -> float:
        """Return current account equity using the most recent prices."""
        return self.balance + sum(
            pos.get("last_price", pos["price"]) * pos["amount"]
            for pos in self.positions.values()
        )

    def _log_to_file(self, entry: Dict) -> None:
        symbol = entry.get("symbol")
        fee = entry.get("fee")
        if not symbol:
            raise ValueError(f"Cannot log trade without symbol: {entry}")
        if fee in (None, ""):
            raise ValueError(f"Cannot log trade without fee: {entry}")

        path = "trade_log.csv"
        file_exists = os.path.isfile(path) and os.path.getsize(path) > 0
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "symbol",
                    "side",
                    "price",
                    "amount",
                    "profit",
                    "fee",
                    "duration",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(entry)

    def buy(
        self,
        price: float,
        amount: float,
        timestamp: pd.Timestamp,
        symbol: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop_pct: Optional[float] = None,
    ) -> bool:
        if not symbol:
            raise ValueError("Symbol must be provided for buy")
        cost = price * amount
        logging.info(
            "Computed buy amount %s for %s at price %.2f (cost %.2f)",
            amount,
            symbol,
            price,
            cost,
        )
        if symbol in self.positions:
            print("Buy skipped: position already open for symbol")
            return False
        if cost > self.initial_balance * self.max_exposure or cost > self.balance:
            print("Buy skipped: exposure limit or insufficient balance")
            return False
        self.balance -= cost
        self.positions[symbol] = {
            "price": price,
            "amount": amount,
            "timestamp": timestamp,
            "symbol": symbol,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "highest_price": price,
            "trailing_stop_pct": trailing_stop_pct,
        }
        self.peak_balance = max(self.peak_balance, self.balance)
        entry = {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "side": "buy",
            "price": price,
            "amount": amount,
            "profit": "",
            "fee": 0.0,
            "duration": "",
        }
        self.log.append(entry)
        self._log_to_file(entry)
        print(
            f"BUY {symbol} {amount} at {price:.2f} -- balance {self.balance:.2f}"
        )
        return True

    def sell(
        self,
        price: float,
        timestamp: pd.Timestamp,
        symbol: str,
        fee_pct: float = 0.0,
        trailing_stop: Optional[float] = None,
    ) -> bool:
        if not symbol:
            raise ValueError("Symbol must be provided for sell")
        pos = self.positions.get(symbol)
        if not pos or pos.get("symbol") != symbol:
            logging.warning(
                "Sell attempted for %s without a matching open position", symbol
            )
            return False
        amount = pos["amount"]
        entry_price = pos["price"]
        exit_price = price
        if trailing_stop is not None and trailing_stop < exit_price:
            exit_price = trailing_stop
        fee = exit_price * amount * fee_pct
        profit = (exit_price - entry_price) * amount - fee
        self.balance += exit_price * amount - fee
        self.peak_balance = max(self.peak_balance, self.balance)
        duration = timestamp - pos["timestamp"]
        entry = {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "side": "sell",
            "price": exit_price,
            "amount": amount,
            "profit": profit,
            "fee": fee,
            "duration": duration.total_seconds(),
        }
        self.log.append(entry)
        self._log_to_file(entry)
        print(
            f"SELL {symbol} {amount} at {exit_price:.2f} -- PnL {profit:.2f} -- balance {self.balance:.2f}"
        )
        del self.positions[symbol]
        self.print_performance()
        return True

    def current_drawdown(
        self, current_prices: Optional[Dict[str, float]] = None
    ) -> float:
        equity = self.balance
        for sym, pos in self.positions.items():
            if current_prices and sym in current_prices:
                price = current_prices[sym]
                pos["last_price"] = price
            else:
                price = pos.get("last_price", pos["price"])
            equity += price * pos["amount"]
        self.peak_balance = max(self.peak_balance, equity)
        return (
            (self.peak_balance - equity) / self.peak_balance
            if self.peak_balance
            else 0.0
        )

    def pnl_by_symbol(self, window: Optional[int] = None) -> Dict[str, float]:
        """Return cumulative PnL for each symbol.

        Parameters
        ----------
        window: int, optional
            Only the last ``window`` closed trades per symbol are included.
        """
        sells = [t for t in self.log if t["side"] == "sell"]
        profits: Dict[str, List[float]] = {}
        for t in sells:
            profits.setdefault(t["symbol"], []).append(float(t["profit"]))
        pnl: Dict[str, float] = {}
        for sym, values in profits.items():
            if window:
                values = values[-window:]
            pnl[sym] = sum(values)
        return pnl

    def print_performance(self) -> None:
        sells = [t for t in self.log if t["side"] == "sell"]
        if not sells:
            return
        equity = self.get_equity()
        net_profit = equity - self.initial_balance
        wins = [t for t in sells if t["profit"] > 0]
        win_rate = len(wins) / len(sells) * 100
        symbol_pnl = self.pnl_by_symbol()
        pnl_str = " | ".join(
            f"{sym}: {pnl:.2f}" for sym, pnl in symbol_pnl.items()
        )
        print(
            f"Trades: {len(sells)} | Net Profit: {net_profit:.2f} | Win rate: {win_rate:.1f}%"
        )
        if pnl_str:
            print(f"PnL by symbol: {pnl_str}")

    def pnl_excluding_outliers(self) -> float:
        """Return PnL with outlier trades removed using the IQR method."""
        sells = [float(t["profit"]) for t in self.log if t["side"] == "sell"]
        if not sells:
            return 0.0
        series = pd.Series(sells)
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        filtered = series[(series >= lower) & (series <= upper)]
        return float(filtered.sum())

    def print_summary(self) -> None:
        """Print a brief account summary."""
        open_trades = len(self.positions)
        closed_trades = len([t for t in self.log if t["side"] == "sell"])
        profit_loss = self.get_equity() - self.initial_balance
        filtered_pnl = self.pnl_excluding_outliers()
        print(
            f"Balance: {self.balance:.2f} | "
            f"Open trades: {open_trades} | "
            f"Closed trades: {closed_trades} | "
            f"PnL: {profit_loss:.2f} | "
            f"PnL (filtered): {filtered_pnl:.2f}"
        )


class TraderBot:
    def __init__(self, config: Config):
        self.config = config
        self.account = PaperAccount(
            config.starting_balance, config.max_exposure, config
        )
        self.symbol_fetcher = SymbolFetcher(min_price=config.min_price)
        self.symbol_fetcher.start()
        self.symbol_fetcher.wait_until_ready()
        self.last_summary = time.time()

    def fetch_candles_ccxt(
        self, exchange_name: str = "binanceus", symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch recent OHLCV data from an exchange via CCXT.

        Parameters
        ----------
        exchange_name: str, optional
            Name of the exchange from the CCXT library. Defaults to ``"binanceus"``.
        symbol: str, optional
            Trading pair in "BASE-QUOTE" format. Defaults to ``self.config.symbol``.
        """
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class()
        pair = (symbol or self.config.symbol).replace("-", "/")
        if pair.endswith("USD"):
            pair = pair[:-3] + "USDT"
        data = exchange.fetch_ohlcv(
            symbol=pair, timeframe=self.config.timeframe, limit=100
        )
        df = pd.DataFrame(
            data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def fetch_candles(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """Fetch recent OHLCV data from the configured exchange via CCXT."""
        try:
            return self.fetch_candles_ccxt(self.config.exchange, symbol)
        except Exception as exc:
            logging.error("CCXT data fetch failed: %s", exc)
            return pd.DataFrame()

    def generate_signal(self, df: pd.DataFrame) -> Optional[str]:
        """Generate a moving average crossover signal filtered by RSI."""
        df["ema_fast"] = df["close"].ewm(span=self.config.ema_fast_span).mean()
        df["ema_slow"] = df["close"].ewm(span=self.config.ema_slow_span).mean()

        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=self.config.rsi_period).mean()
        avg_loss = loss.rolling(window=self.config.rsi_period).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

        rsi_series = df["rsi"]
        rsi_mean = rsi_series.rolling(self.config.rsi_period).mean().iloc[-1]
        rsi_std = rsi_series.rolling(self.config.rsi_period).std().iloc[-1]
        if pd.isna(rsi_mean) or pd.isna(rsi_std):
            buy_thresh = self.config.rsi_buy_threshold
            sell_thresh = self.config.rsi_sell_threshold
        else:
            buy_thresh = max(
                self.config.rsi_buy_threshold,
                rsi_mean + self.config.rsi_std_multiplier * rsi_std,
            )
            sell_thresh = min(
                self.config.rsi_sell_threshold,
                rsi_mean - self.config.rsi_std_multiplier * rsi_std,
            )

        ema_diff = df["ema_fast"] - df["ema_slow"]
        vol = df["close"].pct_change().rolling(self.config.ema_slow_span).std().iloc[-1]
        ema_threshold = (
            vol * self.config.ema_threshold_mult if pd.notna(vol) else 0.0
        )
        if (
            ema_diff.iloc[-1] > ema_threshold
            and ema_diff.iloc[-2] <= ema_threshold
            and df["rsi"].iloc[-1] > buy_thresh
        ):
            return "buy"
        if (
            ema_diff.iloc[-1] < -ema_threshold
            and ema_diff.iloc[-2] >= -ema_threshold
            and df["rsi"].iloc[-1] < sell_thresh
        ):
            return "sell"
        return None

    def execute_trade(
        self, side: str, price: float, timestamp: pd.Timestamp, symbol: str
    ) -> None:
        """Execute a paper trade through the PaperAccount."""
        costs = self.config.fee_pct * 2 + self.config.spread_pct
        if side == "buy":
            atr = None
            if self.config.atr_multiplier > 0:
                df = self.fetch_candles(symbol)
                if not df.empty:
                    high_low = df["high"] - df["low"]
                    high_close = (df["high"] - df["close"].shift()).abs()
                    low_close = (df["low"] - df["close"].shift()).abs()
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr = (
                        tr.rolling(window=self.config.rsi_period, min_periods=1)
                        .mean()
                        .iloc[-1]
                    )
            if symbol not in self.account.positions:
                if atr is not None and atr > 0:
                    target_pct = (atr * self.config.atr_multiplier) / price
                    edge = target_pct - costs
                    stop = price - atr * self.config.atr_multiplier
                    target = price + atr * self.config.atr_multiplier
                else:
                    edge = self.config.take_profit_pct - costs
                    stop = price * (1 - self.config.stop_loss_pct)
                    target = price * (1 + self.config.take_profit_pct)
                if edge < self.config.min_edge_pct:
                    logging.info(
                        "Buy skipped: edge %.4f below minimum %.4f",
                        edge,
                        self.config.min_edge_pct,
                    )
                    return
                amount = self.config.stake_usd / price
                logging.info(
                    "Calculated trade amount %s for stake %.2f at price %.2f",
                    amount,
                    self.config.stake_usd,
                    price,
                )
                if amount > self.config.max_tokens:
                    logging.warning(
                        "Amount %s exceeds max_tokens %s; capping",
                        amount,
                        self.config.max_tokens,
                    )
                    amount = self.config.max_tokens
                if amount <= 0:
                    logging.warning("Buy skipped: non-positive amount %s", amount)
                    return
                self.account.buy(
                    price=price,
                    amount=amount,
                    timestamp=timestamp,
                    symbol=symbol,
                    stop_loss=stop,
                    take_profit=target,
                    trailing_stop_pct=self.config.trailing_stop_pct,
                )
            else:
                logging.info("Buy skipped: position already open for %s", symbol)
        elif side == "sell":
            pos = self.account.positions.get(symbol)
            if pos:
                gain_pct = (price - pos["price"]) / pos["price"] - costs
                if gain_pct < self.config.min_edge_pct:
                    logging.info(
                        "Sell skipped: edge %.4f below minimum %.4f",
                        gain_pct,
                        self.config.min_edge_pct,
                    )
                    return
                self.account.sell(price, timestamp, symbol, self.config.fee_pct)
            else:
                logging.warning(
                    "Sell skipped: no open position for %s", symbol
                )

    def run(self) -> None:
        """Run the trading loop."""
        self.account.print_summary()
        while True:
            symbols = self.symbol_fetcher.symbols or [self.config.symbol]
            paused = False
            for symbol in symbols:
                if not symbol:
                    logging.warning("Encountered empty symbol; skipping trade execution")
                    continue
                pnl = self.account.pnl_by_symbol(self.config.pnl_window).get(symbol, 0.0)
                if pnl < self.config.min_profit_threshold:
                    logging.info(
                        "Skipping %s due to PnL %.2f below threshold %.2f",
                        symbol,
                        pnl,
                        self.config.min_profit_threshold,
                    )
                    continue
                self.config.symbol = symbol
                df = self.fetch_candles()
                if df.empty:
                    time.sleep(1)
                    continue
                price = df["close"].iloc[-1]
                timestamp = df["timestamp"].iloc[-1]
                pos = self.account.positions.get(symbol)
                if pos:
                    pos["last_price"] = price
                    pos["highest_price"] = max(
                        pos.get("highest_price", pos["price"]), price
                    )
                    trail_pct = pos.get("trailing_stop_pct")
                    trailing_stop_price = (
                        pos["highest_price"] * (1 - trail_pct) if trail_pct else None
                    )
                    if (
                        (pos.get("stop_loss") is not None and price <= pos["stop_loss"])
                        or (
                            pos.get("take_profit") is not None
                            and price >= pos["take_profit"]
                        )
                        or (
                            trailing_stop_price is not None
                            and price <= trailing_stop_price
                        )
                    ):
                        self.account.sell(
                            price,
                            timestamp,
                            symbol,
                            self.config.fee_pct,
                            trailing_stop=trailing_stop_price,
                        )
                        continue
                    max_age = pd.Timedelta(minutes=self.config.max_holding_minutes)
                    if timestamp - pos["timestamp"] > max_age:
                        self.account.sell(
                            price, timestamp, symbol, self.config.fee_pct
                        )
                        continue
                signal = self.generate_signal(df)
                if signal == "buy" and not pos:
                    self.execute_trade("buy", price, timestamp, symbol)
                elif signal == "sell" and pos:
                    self.execute_trade("sell", price, timestamp, symbol)
                pos = self.account.positions.get(symbol)
                if pos:
                    pos["last_price"] = price
                if (
                    pos
                    and self.account.current_drawdown({symbol: price})
                    > self.config.max_drawdown_pct
                ):
                    print("Max drawdown exceeded.")
                    exit_df = self.fetch_candles(symbol)
                    if not exit_df.empty:
                        exit_price = exit_df["close"].iloc[-1]
                        exit_time = exit_df["timestamp"].iloc[-1]
                    else:
                        exit_price = price
                        exit_time = timestamp
                    if not self.account.sell(
                        exit_price, exit_time, symbol, self.config.fee_pct
                    ):
                        logging.warning(
                            "Position remains open after drawdown trigger.",
                        )
                    if self.config.stop_on_drawdown:
                        print("Stopping bot due to drawdown.")
                        return
                    self.account.peak_balance = self.account.balance
                    cooldown = self.config.drawdown_cooldown
                    print(f"Pausing for {cooldown} seconds after drawdown.")
                    time.sleep(cooldown)
                    paused = True
                    break
                time.sleep(1)
            if time.time() - self.last_summary >= self.config.summary_interval:
                self.account.print_summary()
                self.last_summary = time.time()
            if not paused:
                time.sleep(60)


if __name__ == "__main__":

    config = Config()

    bot = TraderBot(config)
    try:
        bot.run()
    except Exception as exc:
        logging.exception("Unhandled exception in TraderBot.run: %s", exc)
    finally:
        if bot.account.positions:
            logging.info("Attempting to close open positions on exit.")
            try:
                for pos_symbol in list(bot.account.positions.keys()):
                    df = bot.fetch_candles(pos_symbol)
                    if not df.empty:
                        price = df["close"].iloc[-1]
                        timestamp = df["timestamp"].iloc[-1]
                        if not bot.account.sell(
                            price, timestamp, pos_symbol, bot.config.fee_pct
                        ):
                            logging.warning(
                                "Open position could not be closed on exit.",
                            )
                    else:
                        logging.warning(
                            "No market data to close position on exit; position remains open.",
                        )
            except Exception as exc:
                logging.error("Exception during cleanup: %s", exc)
