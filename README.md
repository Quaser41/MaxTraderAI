# MaxTraderAI

This repository explores building an autonomous cryptocurrency trading bot.

## Overview

The example bot in `src/bot.py` fetches price data from cryptocurrency exchanges via the [`ccxt`](https://github.com/ccxt/ccxt) library (default `binanceus`) and demonstrates a simple moving average crossover strategy. A built-in paper trading account simulates trades with a starting balance of $1000 and a maximum exposure of 75% of the account. Each trade is logged to `trade_log.csv` with profit or loss and the duration the position was held. Stop‑loss and take‑profit levels are applied to every position, and the bot halts if account drawdown exceeds a configurable threshold.


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

To control how often the strategy trades, adjust the `timeframe` along with the `ema_fast_span` and `ema_slow_span` settings in `Config`. Shorter timeframes like `"1m"` or `"5m"` provide more frequent price updates, while smaller EMA spans make crossovers more responsive. These tweaks are useful to trigger trades during brief test runs. For example:

```python
Config(timeframe="1m", ema_fast_span=5, ema_slow_span=15)
```

Use longer spans or higher timeframes for slower, more deliberate trading.

