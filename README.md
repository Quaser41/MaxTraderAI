# MaxTraderAI

This repository explores building an autonomous cryptocurrency trading bot.

## Overview
The example bot in `src/bot.py` fetches price data from [Yahoo Finance](https://finance.yahoo.com/) using the [`yfinance`](https://github.com/ranaroussi/yfinance) library and demonstrates a simple moving average crossover strategy. It logs simulated buy/sell actions rather than placing real trades.

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
