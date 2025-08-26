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
            "profit": 10.0,
            "fee": 0.0,
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
            "profit": -10.0,
            "fee": 0.0,
            "duration": 60.0,
        },
    ]
    df = pd.DataFrame(rows)
    path = tmp_path / "log.csv"
    df.to_csv(path, index=False)
    analyze(str(path))
    captured = capsys.readouterr().out
    assert "Net profit: -2.00" in captured
    assert "Win rate: 50.0% (1/2)" in captured
    assert "AAA-USD: -2.00" in captured


def test_analyze_directory(tmp_path, capsys):
    rows1 = [
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
            "profit": 10.0,
            "fee": 0.0,
            "duration": 60.0,
        },
    ]
    rows2 = [
        {
            "timestamp": "2024-01-02T00:00:00",
            "symbol": "BBB-USD",
            "side": "buy",
            "price": 50.0,
            "amount": 1.0,
            "profit": "",
            "fee": 0.5,
            "duration": "",
        },
        {
            "timestamp": "2024-01-02T00:01:00",
            "symbol": "BBB-USD",
            "side": "sell",
            "price": 40.0,
            "amount": 1.0,
            "profit": -10.0,
            "fee": 0.0,
            "duration": 60.0,
        },
    ]
    df1 = pd.DataFrame(rows1)
    df2 = pd.DataFrame(rows2)
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    df1.to_csv(logs_dir / "log1.csv", index=False)
    df2.to_csv(logs_dir / "log2.csv", index=False)

    analyze(str(logs_dir))
    captured = capsys.readouterr().out
    assert "Net profit: -1.50" in captured
    assert "Win rate: 50.0% (1/2)" in captured
    assert "AAA-USD: 9.00" in captured
    assert "BBB-USD: -10.50" in captured
