import argparse
from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd


def analyze(log_path: str = "trade_log.csv", symbol: Optional[str] = None) -> None:
    """Load a trade log CSV and print summary statistics.

    Parameters
    ----------
    log_path: Path to the CSV trade log.
    symbol: Optional symbol to filter on.
    """
    df = pd.read_csv(log_path, on_bad_lines="skip")
    df = df.dropna(subset=["symbol"])  # ignore malformed rows
    if symbol:
        df = df[df["symbol"] == symbol]
    if df.empty:
        print("No trades found")
        return
    df = df.sort_values("timestamp")
    if "fee" not in df.columns:
        df["fee"] = 0.0
    else:
        df["fee"] = df["fee"].fillna(0.0)
    if "profit" not in df.columns:
        df["profit"] = 0.0
    else:
        df["profit"] = df["profit"].fillna(0.0)

    buy_fees: Dict[str, List[float]] = defaultdict(list)
    per_symbol: Dict[str, float] = defaultdict(float)
    wins = 0
    trades = 0
    total_profit = 0.0

    for _, row in df.iterrows():
        side = row["side"]
        sym = row["symbol"]
        if side == "buy":
            buy_fees[sym].append(float(row["fee"]))
        elif side == "sell":
            fee = buy_fees[sym].pop(0) if buy_fees[sym] else 0.0
            net = float(row["profit"]) - fee
            total_profit += net
            per_symbol[sym] += net
            trades += 1
            if net > 0:
                wins += 1

    win_rate = 100 * wins / trades if trades else 0.0
    print(f"Net profit: {total_profit:.2f}")
    print(f"Win rate: {win_rate:.1f}% ({wins}/{trades})")
    if per_symbol:
        print("Per-symbol PnL:")
        for sym, pnl in per_symbol.items():
            print(f"  {sym}: {pnl:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a trade log CSV")
    parser.add_argument("--path", default="trade_log.csv", help="Path to trade log")
    parser.add_argument("--symbol", help="Filter by symbol")
    args = parser.parse_args()
    analyze(args.path, symbol=args.symbol)


if __name__ == "__main__":
    main()
