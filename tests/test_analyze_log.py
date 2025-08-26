import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from analyze_log import analyze


def test_analyze_basic(tmp_path, capsys):
    rows = [
        {
            "timestamp": "2024-01-01T00:00:00",
            "symbol": "AAA-USD",
            "side": "buy",
            "price": 100.0,
            "amount": 1.0,
            "profit": "",
            "fee": 1.0,
            "duration": "",
        },
        {
            "timestamp": "2024-01-01T00:01:00",
            "symbol": "AAA-USD",
            "side": "sell",
            "price": 110.0,
            "amount": 1.0,
            "profit": 8.5,
            "fee": 1.5,
            "duration": 60.0,
        },
        {
            "timestamp": "2024-01-01T00:02:00",
            "symbol": "AAA-USD",
            "side": "buy",
            "price": 100.0,
            "amount": 1.0,
            "profit": "",
            "fee": 1.0,
            "duration": "",
        },
        {
            "timestamp": "2024-01-01T00:03:00",
            "symbol": "AAA-USD",
            "side": "sell",
            "price": 90.0,
            "amount": 1.0,
            "profit": -11.5,
            "fee": 1.5,
            "duration": 60.0,
        },
    ]
    df = pd.DataFrame(rows)
    path = tmp_path / "log.csv"
    df.to_csv(path, index=False)
    analyze(str(path))
    captured = capsys.readouterr().out
    assert "Net profit: -3.00" in captured
    assert "Win rate: 50.0% (1/2)" in captured
    assert "AAA-USD: -3.00" in captured
