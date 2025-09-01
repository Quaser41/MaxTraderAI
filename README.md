# MaxTraderAI

This repository explores building an autonomous cryptocurrency trading bot.

## Overview

The example bot in `src/bot.py` fetches price data from cryptocurrency exchanges via the [`ccxt`](https://github.com/ccxt/ccxt) library (default `binanceus`) and demonstrates a simple moving average crossover strategy. A built-in paper trading account simulates trades with a starting balance of $1000 and a maximum exposure of 75% of the account. Each trade is logged to `trade_log.csv` with profit or loss and the duration the position was held. Stop‑loss and take‑profit levels are applied to every position, and the bot halts if account drawdown exceeds a configurable threshold.

By default the bot uses a 5‑minute timeframe with 20/50 EMA spans and RSI thresholds of 60/40. This slower configuration reduces trade frequency and may improve win rate by filtering out some market noise.


## Disclaimer
This project is for educational purposes only and does not constitute financial or investment advice. Use at your own risk.

## Installation
Install dependencies, including `ccxt`, with:

```bash
pip install -r requirements.txt
```

## Usage

- On any platform, run:
  ```bash
  python src/bot.py
  ```
- On Windows, `start_bot.bat` installs dependencies and launches the bot.
- Use `update.bat` to pull the latest repository changes.

Configuration such as trading pair, exchange (default `binanceus`), stop‑loss/take‑profit percentages, drawdown limit, starting balance, and exposure limits can be adjusted in the `Config` dataclass inside `src/bot.py`. The exchange value determines which CCXT exchange provides price data.

Set `debug_logging=True` in `Config` to enable additional `logging.debug` messages that explain when RSI/EMA conditions fail or when orders are skipped because of edge, PnL threshold, or exposure limits.


`stake_usd` and `risk_pct` set trade size; at least one of them must be greater than zero. `max_tokens` also needs to be a positive number. The bot validates these minimum values on startup and raises an error if the resulting trade size is not positive.

Because every trade pays fees and crosses the bid/ask spread, the bot only
enters positions when the profit target exceeds these costs plus a minimum
required edge. Total trading costs are computed as `fee_pct*2 + spread_pct`
and compared against `take_profit_pct - min_edge_pct`. If the target is too
small to cover costs and the desired edge, the bot logs a warning and skips
trades.


The bot tracks profit and loss by symbol. Use `pnl_window` to set the number of recent closed trades to evaluate and `min_profit_threshold` (default `0.1`) to require a minimum cumulative profit before continuing to trade a symbol. A symbol must earn at least this amount over the configured window before further trades are allowed; otherwise it is skipped. Set the threshold to `0` to disable this check.

The strategy also calculates a 14-period Relative Strength Index (RSI) and only allows a trade when this value is above `rsi_buy_threshold` (default 60) for buys or below `rsi_sell_threshold` (default 40) for sells. The RSI period and thresholds are configurable via `rsi_period`, `rsi_buy_threshold`, and `rsi_sell_threshold` in `Config`. Typical RSI thresholds range from about 55–70 for buys and 30–45 for sells. Lowering the buy threshold or raising the sell threshold increases signal frequency, which can help when running with tight profit targets.

To control how often the strategy trades, adjust the `timeframe` along with the `ema_fast_span` and `ema_slow_span` settings in `Config`. Shorter timeframes like `"1m"` provide more frequent price updates, while smaller EMA spans make crossovers more responsive. Common spans are 5‑20 for the fast EMA and 20‑100 for the slow EMA. The default 5‑minute timeframe with 20/50 spans trades less often, which may improve win rate. These tweaks are useful to trigger trades during brief test runs, and tighter profit targets may require shorter spans or lower thresholds to ensure enough signals. For example:

```python
Config(timeframe="1m", ema_fast_span=5, ema_slow_span=15)
```

Use longer spans or higher timeframes for slower, more deliberate trading.

