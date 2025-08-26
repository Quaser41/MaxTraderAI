import argparse
from collections import defaultdict
from typing import Dict, List, Optional
import os


import pandas as pd


def analyze(log_path: str = "trade_log.csv", symbol: Optional[str] = None) -> None:
    """Load a trade log CSV and print summary statistics.

    Parameters
    ----------
    log_path: Path to the CSV trade log or directory of logs.
    symbol: Optional symbol to filter on.
    """
    paths: List[str]
    if os.path.isdir(log_path):
        paths = [
            os.path.join(log_path, p)
            for p in os.listdir(log_path)
            if p.endswith(".csv")
        ]
    else:
        paths = [log_path]

    dfs = []
    for p in paths:
        try:
            dfs.append(pd.read_csv(p, on_bad_lines="skip"))
        except FileNotFoundError:
            continue
    if not dfs:
        print("No trades found")
        return

    df = pd.concat(dfs, ignore_index=True)
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

    sells = df[df["side"] == "sell"]
    profits = sells["profit"].astype(float)
    total_profit = float(profits.sum())
    trades = len(profits)
    wins = int((profits > 0).sum())
    win_rate = 100 * wins / trades if trades else 0.0
    per_symbol = (
        sells.groupby("symbol")["profit"].sum().astype(float).to_dict()
        if not sells.empty
        else {}
    )

    print(f"Net profit: {total_profit:.2f}")
    print(f"Win rate: {win_rate:.1f}% ({wins}/{trades})")
    if per_symbol:
        print("Per-symbol PnL:")
        for sym, pnl in per_symbol.items():
            print(f"  {sym}: {pnl:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze trade log CSV files")
    parser.add_argument(
        "--path", default="trade_log.csv", help="Path to trade log file or directory"
    )
    parser.add_argument("--symbol", help="Filter by symbol")
    args = parser.parse_args()
    analyze(args.path, symbol=args.symbol)


if __name__ == "__main__":
    main()
