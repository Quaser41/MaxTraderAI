import sys
from pathlib import Path
import time
import requests
import ccxt
import pytest

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bot import Config, SymbolFetcher


def test_symbol_fetcher_filters_low_price(monkeypatch):
    ticker_data = [
        {"symbol": "CHEAPUSDT", "quoteVolume": "2000000", "lastPrice": "0.5"},
        {"symbol": "NORMALUSDT", "quoteVolume": "1000000", "lastPrice": "2.0"},
    ]

    class DummyResponse:
        def json(self):
            return ticker_data

    class DummyExchange:
        def load_markets(self):
            pass

        def fetch_ticker(self, pair):
            return {"symbol": pair}

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: DummyResponse())
    monkeypatch.setattr(ccxt, "binanceus", lambda: DummyExchange())
    monkeypatch.setattr(time, "sleep", lambda _: (_ for _ in ()).throw(SystemExit))

    config = Config(min_price=1.0)
    fetcher = SymbolFetcher(min_price=config.min_price)
    with pytest.raises(SystemExit):
        fetcher._run()

    assert "CHEAP-USD" not in fetcher.symbols
    assert "NORMAL-USD" in fetcher.symbols
