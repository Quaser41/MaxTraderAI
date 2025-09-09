import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import bot as bot_module
from bot import Config, TraderBot


class DummyFetcher:
    symbols = []

    def start(self):
        pass

    def wait_until_ready(self, timeout=None):
        pass


def test_reset_pnl(monkeypatch):
    # Avoid network calls by stubbing out SymbolFetcher
    monkeypatch.setattr(
        bot_module,
        "SymbolFetcher",
        lambda min_price=0.0, fallback_symbols=None: DummyFetcher(),
    )

    cfg = Config(min_profit_threshold=0.1)
    bot = TraderBot(cfg)
    bot._pnl_offsets["AAA-USD"] = -5.0
    bot._pnl_block_until["AAA-USD"] = 123.0

    bot.reset_pnl()

    assert bot._pnl_offsets == {}
    assert bot._pnl_block_until == {}
