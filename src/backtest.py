import argparse
from dataclasses import fields
from typing import Any

import ccxt
import pandas as pd

from bot import Config, PaperAccount, TraderBot


class Backtester:
    """Simple backtesting utility using :class:`PaperAccount` and signals from
    :class:`TraderBot`.

    The class reuses the ``generate_signal`` logic from ``TraderBot`` without
    starting background threads or performing live trading.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.account = PaperAccount(
            balance=config.starting_balance,
            max_exposure=config.max_exposure,
            config=config,
        )

    def generate_signal(self, df: pd.DataFrame) -> str | None:
        """Proxy to :meth:`TraderBot.generate_signal`."""
        return TraderBot.generate_signal(self, df)

    def fetch_candles(self, limit: int) -> pd.DataFrame:
        """Fetch historical candles via CCXT for the configured symbol."""
        exchange_class = getattr(ccxt, self.config.exchange)
        exchange = exchange_class()
        pair = self.config.symbol.replace("-", "/")
        if pair.endswith("USD"):
            pair = pair[:-3] + "USDT"
        data = exchange.fetch_ohlcv(pair, timeframe=self.config.timeframe, limit=limit)
        df = pd.DataFrame(
            data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def run(self, limit: int) -> None:
        """Execute the backtest and print performance metrics."""
        df = self.fetch_candles(limit)
        if df.empty:
            print("No candle data returned from exchange")
            return
        lookback = max(self.config.ema_slow_span, self.config.rsi_period) + 1
        symbol = self.config.symbol
        for i in range(lookback, len(df)):
            window = df.iloc[: i + 1].copy()
            signal = self.generate_signal(window)
            price = window["close"].iloc[-1]
            timestamp = window["timestamp"].iloc[-1]
            pos = self.account.positions.get(symbol)
            if pos:
                pos["last_price"] = price
            if signal == "buy":
                amount = self.config.stake_usd / price if price > 0 else 0.0
                if amount > self.config.max_tokens:
                    amount = self.config.max_tokens
                if amount > 0:
                    self.account.buy(
                        price, amount, timestamp, symbol, fee_pct=self.config.fee_pct
                    )
            elif signal == "sell":
                self.account.sell(price, timestamp, symbol, fee_pct=self.config.fee_pct)

        # Close any open position at final price
        if symbol in self.account.positions:
            last = df.iloc[-1]
            self.account.sell(
                last["close"], last["timestamp"], symbol, fee_pct=self.config.fee_pct
            )
        self.account.print_performance()


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse CLI ``--param key=value`` overrides using ``Config`` types."""
    overrides: dict[str, Any] = {}
    field_types = {f.name: f.type for f in fields(Config)}
    for pair in pairs:
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        if key not in field_types:
            raise ValueError(f"Unknown configuration field: {key}")
        ftype = field_types[key]
        if ftype is bool:
            parsed = value.lower() in {"1", "true", "yes", "on"}
        else:
            parsed = ftype(value)
        overrides[key] = parsed
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest trading strategy")
    parser.add_argument("--symbol", default="BTC-USD", help="Trading pair symbol")
    parser.add_argument("--timeframe", default="5m", help="Candle timeframe")
    parser.add_argument("--limit", type=int, default=500, help="Number of candles")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Override Config parameter, e.g. --param ema_fast_span=10",
    )
    args = parser.parse_args()

    cfg_kwargs = {"symbol": args.symbol, "timeframe": args.timeframe}
    cfg_kwargs.update(parse_overrides(args.param))
    config = Config(**cfg_kwargs)
    backtester = Backtester(config)
    backtester.run(limit=args.limit)


if __name__ == "__main__":
    main()
