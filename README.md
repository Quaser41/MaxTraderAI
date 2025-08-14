# MaxTraderAI

This repository explores building an autonomous cryptocurrency trading bot.

## Overview
The example bot in `src/bot.py` connects to an exchange using the [CCXT](https://github.com/ccxt/ccxt) library and trades a simple moving average crossover strategy.

## Disclaimer
This project is for educational purposes only and does not constitute financial or investment advice. Use at your own risk.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
Adjust the configuration in `src/bot.py` with your exchange API credentials, then run:
```bash
python src/bot.py
```
