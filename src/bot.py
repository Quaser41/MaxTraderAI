"""

Minimal autonomous crypto trading bot using CCXT data.


This code is for educational purposes only and does not constitute financial advice.
"""
from dataclasses import dataclass
from typing import Optional, List, Dict
import time
import csv
import os
import shutil
from datetime import datetime
import logging
import threading
import argparse
import json
import importlib
import pandas as pd
import requests
import ccxt

logging.basicConfig(level=logging.INFO)


@dataclass
class Config:
    """Trading parameters for :class:`TraderBot`.

    ``take_profit_pct`` must exceed the total trading costs plus a minimum
    required edge for a position to be considered. The costs combine entry and
    exit fees with the estimated bid/ask spread and are computed as
    ``fee_pct * 2 + spread_pct``. If ``take_profit_pct`` is less than or equal
    to ``costs + min_edge_pct``, the bot will skip trades because the potential
    profit cannot cover expenses and the desired edge.
    """

    symbol: str = "BTC-USD"
    timeframe: str = "5m"
    exchange: str = "binanceus"
    strategy: str = "ema_rsi"  # name of strategy module under strategies package
    stake_usd: float = 100.0  # trade size in USD (must be > 0 if used)
    risk_pct: float = 0.01  # fraction of equity to risk per trade (must be > 0 if used)
    max_tokens: float = float("inf")  # maximum token quantity per trade (must be > 0)
    starting_balance: float = 1000.0
    max_exposure: float = 0.75  # fraction of account allowed in a single trade
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.04  # 4% take profit target
    atr_multiplier: float = 1.0  # ATR multiple for stop-loss and take-profit
    max_drawdown_pct: float = 0.2  # stop trading if drawdown exceeds 20%
    ema_fast_span: int = 20  # fast EMA span for crossover (5-20 typical; shorten for more signals) --override with --ema-fast-span
    ema_slow_span: int = 50  # slow EMA span for crossover (20-100 typical; shorten with tight targets) --override with --ema-slow-span
    drawdown_cooldown: int = 300  # seconds to pause after max drawdown
    stop_on_drawdown: bool = True  # stop bot instead of pausing on drawdown
    summary_interval: int = 300  # seconds between status summaries
    pnl_window: int = 10  # number of closed trades to evaluate per-symbol PnL
    min_profit_threshold: float = 0.0  # minimum profit to keep trading a symbol (<=0 disables)
    pnl_cooldown: int = 0  # seconds to ignore a symbol after failing the PnL filter
    fee_pct: float = 0.001  # exchange fee percentage applied on sells
    trailing_stop_pct: float = 0.01  # percentage for trailing stop (0 to disable)
    max_holding_minutes: int = 60  # maximum duration to hold a position
    rsi_period: int = 14  # period for RSI calculation
    rsi_buy_threshold: float = 60.0  # minimum RSI for buy signals (55-70 typical; lower for tight targets) --override with --rsi-buy-threshold
    rsi_sell_threshold: float = 40.0  # maximum RSI for sell signals (30-45 typical; raise for tight targets) --override with --rsi-sell-threshold
    rsi_std_multiplier: float = 1.0  # std-dev multiplier for adaptive RSI
    use_rsi_filter: bool = True  # apply RSI threshold checks to signals
    ema_threshold_mult: float = 0.0  # volatility factor for EMA crossover
    spread_pct: float = 0.0005  # estimated bid/ask spread percentage
    min_edge_pct: float = 0.0015  # minimum edge required after costs
    min_price: float = 0.0  # minimum token price to include
    debug_logging: bool = False  # enable detailed debug logs
    symbols: Optional[List[str]] = None  # fallback symbols if auto-fetch fails




class SymbolFetcher:
    """Background thread that refreshes top-volume symbols from BinanceUS."""

    def __init__(
        self,
        refresh: int = 3600,
        limit: int = 10,
        min_price: float = 0.0,
        fallback_symbols: Optional[List[str]] = None,
    ) -> None:
        self.refresh = refresh
        self.limit = limit
        self.min_price = min_price
        self.symbols: List[str] = []
        self.fallback_symbols = fallback_symbols or []
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
        else:
            logging.warning("Symbol fetcher did not return any symbols")

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
                else:
                    logging.warning("No symbols fetched; will retry")
                    if self.fallback_symbols and not self._ready.is_set():
                        logging.warning(
                            "Using fallback symbols: %s",
                            ", ".join(self.fallback_symbols),
                        )
                        self.symbols = list(self.fallback_symbols)
                        self._ready.set()
            except Exception as exc:
                logging.error("Symbol fetch failed: %s", exc)
                if self.fallback_symbols and not self.symbols:
                    logging.warning(
                        "Using fallback symbols due to fetch failure: %s",
                        ", ".join(self.fallback_symbols),
                    )
                    self.symbols = list(self.fallback_symbols)
                    if not self._ready.is_set():
                        self._ready.set()
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

        # log file management
        self.log_dir = "logs"
        self.log_file = os.path.join(self.log_dir, "trade_log.csv")
        self._prepare_log_file()

    def get_equity(self) -> float:
        """Return current account equity using the most recent prices."""
        return self.balance + sum(
            pos.get("last_price", pos["price"]) * pos["amount"]
            for pos in self.positions.values()
        )

    def _prepare_log_file(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        # move any existing log file to a timestamped name
        legacy_paths = ["trade_log.csv", self.log_file]
        for path in legacy_paths:
            if os.path.exists(path):
                date_str = datetime.now().strftime("%Y%m%d")
                dest = os.path.join(self.log_dir, f"trade_log_{date_str}.csv")
                counter = 1
                while os.path.exists(dest):
                    dest = os.path.join(
                        self.log_dir, f"trade_log_{date_str}_{counter}.csv"
                    )
                    counter += 1
                shutil.move(path, dest)
                break

        # always create a fresh log file with header so it exists even with no trades
        header = [
            "timestamp",
            "symbol",
            "side",
            "price",
            "amount",
            "profit",
            "fee",
            "duration",
        ]
        with open(self.log_file, "w", newline="") as f:
            csv.writer(f).writerow(header)

    def _log_to_file(self, entry: Dict) -> None:
        symbol = entry.get("symbol")
        if not symbol:
            raise ValueError(f"Cannot log trade without symbol: {entry}")

        os.makedirs(self.log_dir, exist_ok=True)
        path = self.log_file
        header = [
            "timestamp",
            "symbol",
            "side",
            "price",
            "amount",
            "profit",
            "fee",
            "duration",
        ]
        file_exists = os.path.isfile(path)
        if file_exists:
            with open(path, newline="") as f:
                reader = csv.reader(f)
                first_row = next(reader, [])
            if first_row != header:
                with open(path, "w", newline="") as f:
                    csv.writer(f).writerow(header)
                file_exists = True

        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerow(entry)

    def buy(
        self,
        price: float,
        amount: float,
        timestamp: pd.Timestamp,
        symbol: str,
        fee_pct: float = 0.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop_pct: Optional[float] = None,
    ) -> bool:
        """Open a position and record optional trailing-stop settings."""
        if not symbol:
            raise ValueError("Symbol must be provided for buy")
        spread_pct = self.config.spread_pct
        effective_price = price * (1 + spread_pct / 2)
        cost = effective_price * amount
        fee = cost * fee_pct
        total_cost = cost + fee
        logging.info(
            "Computed buy amount %s for %s at price %.2f (cost %.2f)",
            amount,
            symbol,
            effective_price,
            cost,
        )
        if symbol in self.positions:
            print("Buy skipped: position already open for symbol")
            return False
        if (
            total_cost > self.get_equity() * self.max_exposure
            or total_cost > self.balance
        ):
            print("Buy skipped: exposure limit or insufficient balance")
            return False
        self.balance -= total_cost
        self.positions[symbol] = {
            "price": effective_price,
            "amount": amount,
            "timestamp": timestamp,
            "symbol": symbol,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "highest_price": effective_price,
            "trailing_stop_pct": trailing_stop_pct,
            "buy_fee": fee,
        }
        self.peak_balance = max(self.peak_balance, self.balance)
        entry = {
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "side": "buy",
            "price": effective_price,
            "amount": amount,
            "profit": "",
            "fee": fee,
            "duration": "",
        }
        self.log.append(entry)
        self._log_to_file(entry)
        print(
            f"BUY {symbol} {amount} at {effective_price:.2f} -- balance {self.balance:.2f}"
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
        """Close a position, optionally respecting a trailing-stop price."""
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
        exit_price = price * (1 - self.config.spread_pct / 2)
        if trailing_stop is not None and trailing_stop <= exit_price:
            exit_price = trailing_stop * (1 - self.config.spread_pct / 2)
        fee = exit_price * amount * fee_pct
        buy_fee = float(pos.get("buy_fee", 0.0))
        total_fee = fee + buy_fee
        profit = (exit_price - entry_price) * amount - total_fee
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
            "fee": total_fee,
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
        if self.config.debug_logging:
            logging.getLogger().setLevel(logging.DEBUG)
        try:
            self.strategy = importlib.import_module(
                f"strategies.{self.config.strategy}"
            )
        except Exception as exc:  # pragma: no cover - config errors
            raise ValueError(
                f"Could not load strategy '{self.config.strategy}': {exc}"
            ) from exc
        self.costs = config.fee_pct * 2 + config.spread_pct
        if config.take_profit_pct <= self.costs + config.min_edge_pct:
            logging.warning(
                "take_profit_pct %.4f is <= costs %.4f + min_edge_pct %.4f; trades will be skipped due to insufficient edge",
                config.take_profit_pct,
                self.costs,
                config.min_edge_pct,
            )
        self.account = PaperAccount(
            config.starting_balance, config.max_exposure, config
        )
        self._validate_trade_size()
        self.symbol_fetcher = SymbolFetcher(
            min_price=config.min_price, fallback_symbols=config.symbols
        )
        self.symbol_fetcher.start()
        self.symbol_fetcher.wait_until_ready()
        if not self.symbol_fetcher.symbols:
            if config.symbols:
                logging.warning(
                    "Symbol fetcher returned no symbols; using fallback list: %s",
                    ", ".join(config.symbols),
                )
                self.symbol_fetcher.symbols = list(config.symbols)
            else:
                logging.warning(
                    "Symbol fetcher returned no symbols; defaulting to %s",
                    config.symbol,
                )
                self.symbol_fetcher.symbols = [config.symbol]
        self.last_summary = time.time()
        self._pnl_block_until: Dict[str, float] = {}
        self._pnl_offsets: Dict[str, float] = {}

    def reset_pnl(self) -> None:
        """Clear per-symbol PnL filters so all symbols are eligible for trading."""
        self._pnl_offsets.clear()
        self._pnl_block_until.clear()

    def _validate_trade_size(self) -> None:
        """Ensure configuration results in a positive trade size."""
        price = 1.0  # use nominal price; ratios are price-invariant
        stake_amount = self.config.stake_usd / price if price > 0 else 0.0
        equity = self.account.initial_balance
        stop_distance = price * (
            self.config.stop_loss_pct + self.config.spread_pct / 2
        )
        risk_amount = (
            equity * self.config.risk_pct / stop_distance
            if stop_distance > 0
            else 0.0
        )
        stake_amount = min(stake_amount, self.config.max_tokens)
        risk_amount = min(risk_amount, self.config.max_tokens)
        if stake_amount <= 0 and risk_amount <= 0:
            raise ValueError(
                "Configuration results in non-positive trade size; "
                "check stake_usd, risk_pct, and max_tokens."
            )

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
        """Delegate to the configured strategy module."""
        return self.strategy.generate_signal(df, self.config)

    def execute_trade(
        self, side: str, price: float, timestamp: pd.Timestamp, symbol: str
    ) -> None:
        """Execute a paper trade through the PaperAccount."""
        costs = self.costs
        max_capital = self.account.balance * self.config.max_exposure
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
                    logging.debug(
                        "Buy skipped: edge %.4f below minimum %.4f",
                        edge,
                        self.config.min_edge_pct,
                    )
                    return
                if self.config.risk_pct > 0 and stop < price:
                    equity = self.account.get_equity()
                    risk_amount = equity * self.config.risk_pct
                    entry_price = price * (1 + self.config.spread_pct / 2)
                    stop_distance = entry_price - stop
                    if stop_distance <= 0:
                        logging.warning(
                            "Buy skipped: non-positive stop distance (entry %.2f, stop %.2f)",
                            entry_price,
                            stop,
                        )
                        return
                    amount = risk_amount / stop_distance
                    cost = amount * entry_price
                    if cost > max_capital:
                        logging.info(
                            "Risk-based amount cost %.2f exceeds max exposure %.2f",
                            cost,
                            max_capital,
                        )
                        stake_amount = self.config.stake_usd / price if price > 0 else 0.0
                        stake_cost = stake_amount * entry_price
                        if stake_amount > 0 and stake_cost <= max_capital:
                            amount = stake_amount
                            logging.info(
                                "Using stake-based amount %s due to capital limit",
                                amount,
                            )
                        else:
                            amount = max_capital / entry_price if entry_price > 0 else 0.0
                            logging.info(
                                "Capping amount to %s due to capital limit",
                                amount,
                            )
                    actual_risk = amount * stop_distance
                    logging.info(
                        "Calculated trade amount %s risking %.2f (equity %.2f * risk_pct %.4f) with stop %.2f",
                        amount,
                        actual_risk,
                        equity,
                        self.config.risk_pct,
                        stop,
                    )
                else:
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
                    logging.debug(
                        "Buy skipped due to exposure limit %.2f",
                        max_capital,
                    )
                    return
                self.account.buy(
                    price=price,
                    amount=amount,
                    timestamp=timestamp,
                    symbol=symbol,
                    fee_pct=self.config.fee_pct,
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
                    logging.debug(
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
            if not self.symbol_fetcher.symbols:
                if self.config.symbols:
                    logging.warning(
                        "Symbol list empty; falling back to user-supplied list: %s",
                        ", ".join(self.config.symbols),
                    )
                    symbols = self.config.symbols
                else:
                    logging.warning(
                        "Symbol list empty; using configured symbol %s",
                        self.config.symbol,
                    )
                    symbols = [self.config.symbol]
            else:
                symbols = self.symbol_fetcher.symbols
            paused = False
            for symbol in symbols:
                if not symbol:
                    logging.warning("Encountered empty symbol; skipping trade execution")
                    continue
                if self.config.min_profit_threshold > 0:
                    now = time.time()
                    block_until = self._pnl_block_until.get(symbol, 0)
                    if block_until > now:
                        logging.debug(
                            "Skipping %s during PnL cooldown for %.0f more seconds",
                            symbol,
                            block_until - now,
                        )
                        continue
                    pnl_values = self.account.pnl_by_symbol(self.config.pnl_window)
                    raw_pnl = pnl_values.get(symbol, 0.0)
                    if block_until and block_until <= now:
                        self._pnl_offsets[symbol] = raw_pnl
                        self._pnl_block_until.pop(symbol, None)
                    adjusted_pnl = raw_pnl - self._pnl_offsets.get(symbol, 0.0)
                    if (
                        adjusted_pnl < self.config.min_profit_threshold
                        and adjusted_pnl != 0
                    ):
                        logging.debug(
                            "Skipping %s due to PnL %.2f below threshold %.2f",
                            symbol,
                            adjusted_pnl,
                            self.config.min_profit_threshold,
                        )
                        if self.config.pnl_cooldown > 0:
                            self._pnl_block_until[symbol] = now + self.config.pnl_cooldown
                            self._pnl_offsets[symbol] = raw_pnl
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

    parser = argparse.ArgumentParser(description="Run TraderBot")
    parser.add_argument("--config", type=str, help="Path to JSON config file", default=None)
    parser.add_argument(
        "--min-profit-threshold",
        type=float,
        help="Override minimum profit threshold (<=0 disables)",
        default=None,
    )
    parser.add_argument(
        "--reset-pnl",
        action="store_true",
        help="Reset per-symbol PnL filters and cooldowns",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated fallback symbols if auto-fetch fails",
        default=None,
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        help="Override timeframe for candle data (e.g., 1m, 5m)",
        default=None,
    )
    parser.add_argument(
        "--ema-fast-span",
        type=int,
        help="Override fast EMA span",
        default=None,
    )
    parser.add_argument(
        "--ema-slow-span",
        type=int,
        help="Override slow EMA span",
        default=None,
    )
    parser.add_argument(
        "--rsi-buy-threshold",
        type=float,
        help="Override RSI buy threshold",
        default=None,
    )
    parser.add_argument(
        "--rsi-sell-threshold",
        type=float,
        help="Override RSI sell threshold",
        default=None,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy module name (e.g., ema_rsi, rsi_mean)",
        default=None,
    )
    parser.add_argument(
        "--min-edge-pct",
        type=float,
        help="Override minimum edge required after costs",
        default=None,
    )
    parser.add_argument(
        "--no-rsi-filter",
        action="store_false",
        dest="use_rsi_filter",
        help="Disable RSI threshold checks",
        default=None,
    )
    args = parser.parse_args()

    cfg_kwargs = {}
    if args.config:
        try:
            with open(args.config) as f:
                cfg_kwargs.update(json.load(f))
        except Exception as exc:  # pragma: no cover - config loading errors are logged
            logging.error("Failed to load config file %s: %s", args.config, exc)
    if args.min_profit_threshold is not None:
        cfg_kwargs["min_profit_threshold"] = args.min_profit_threshold
    if args.timeframe is not None:
        cfg_kwargs["timeframe"] = args.timeframe
    if args.ema_fast_span is not None:
        cfg_kwargs["ema_fast_span"] = args.ema_fast_span
    if args.ema_slow_span is not None:
        cfg_kwargs["ema_slow_span"] = args.ema_slow_span
    if args.rsi_buy_threshold is not None:
        cfg_kwargs["rsi_buy_threshold"] = args.rsi_buy_threshold
    if args.rsi_sell_threshold is not None:
        cfg_kwargs["rsi_sell_threshold"] = args.rsi_sell_threshold
    if args.strategy is not None:
        cfg_kwargs["strategy"] = args.strategy
    if args.min_edge_pct is not None:
        cfg_kwargs["min_edge_pct"] = args.min_edge_pct
    if args.use_rsi_filter is not None:
        cfg_kwargs["use_rsi_filter"] = args.use_rsi_filter
    if args.symbols is not None:
        cfg_kwargs["symbols"] = [s.strip() for s in args.symbols.split(",") if s.strip()]

    config = Config(**cfg_kwargs)

    bot = TraderBot(config)
    if args.reset_pnl:
        bot.reset_pnl()
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
