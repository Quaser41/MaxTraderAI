# MaxTraderAI

This repository explores building an autonomous cryptocurrency trading bot.

## Overview

The example bot in `src/bot.py` fetches price data from [Yahoo Finance](https://finance.yahoo.com/) using the [`yfinance`](https://github.com/ranaroussi/yfinance) library and demonstrates a simple moving average crossover strategy. A built-in paper trading account simulates trades with a starting balance of $1000 and a maximum exposure of 75% of the account. Each trade is logged to `trade_log.csv` with profit or loss and the duration the position was held.


## Disclaimer
This project is for educational purposes only and does not constitute financial or investment advice. Use at your own risk.

## Installation
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

Configuration such as trading pair, starting balance, and exposure limits can be adjusted in the `Config` dataclass inside `src/bot.py`.

