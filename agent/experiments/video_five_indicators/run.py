"""Run a no-order, auditable three-month Vibe Trading crypto backtest.

Run from agent/: python -m experiments.video_five_indicators.run
Official Binance USD-M historical CSVs; no exchange credentials required.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from backtest.engines.crypto import CryptoEngine
from experiments.video_five_indicators.signal_engine import SignalEngine

SOURCE = "https://data.binance.vision/data/futures/um"
EVAL_START = "2026-08-12"
EVAL_END_EXCLUSIVE = "2026-11-12"
WARMUP_START = "2026-07-01"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def archive_urls(symbol: str, timeframe: str) -> list[str]:
    monthly = [
        f"{SOURCE}/monthly/klines/{symbol}/{timeframe}/{symbol}-{timeframe}-2026-{month:02d}.zip"
        for month in range(7, 11)
    ]
    daily = [
        f"{SOURCE}/daily/klines/{symbol}/{timeframe}/{symbol}-{timeframe}-2026-11-{day:02d}.zip"
        for day in range(1, 12)
    ]
    return monthly + daily


def download_zip(url: str, session: requests.Session) -> bytes:
    for attempt in range(3):
        try:
            response = session.get(url, timeout=(10, 45))
            response.raise_for_status()
            data = response.content
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                if not any(name.endswith(".csv") for name in zf.namelist()):
                    raise ValueError(f"Missing CSV in {url}")
            return data
        except (requests.RequestException, zipfile.BadZipFile, ValueError) as exc:
            if attempt == 2:
                raise RuntimeError(f"Data unavailable: {url}: {exc}") from exc
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def read_archive(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        name = next(name for name in zf.namelist() if name.endswith(".csv"))
        raw = pd.read_csv(io.BytesIO(zf.read(name)), header=None, low_memory=False)
    frame = raw.iloc[:, :6].copy()
    frame.columns = ["time", "open", "high", "low", "close", "volume"]
    for name in frame.columns:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.dropna(subset=list(frame.columns))
    ts = frame.pop("time").astype("int64")
    unit = "us" if ts.median() > 100_000_000_000_000 else "ms"
    frame.index = pd.to_datetime(ts, unit=unit, utc=True)
    frame.index.name = "timestamp"
    return frame.astype(float)


def historical_data(
    symbol: str, timeframe: str, session: requests.Session
) -> tuple[pd.DataFrame, list[dict]]:
    frames, provenance = [], []
    for url in archive_urls(symbol, timeframe):
        data = download_zip(url, session)
        frame = read_archive(data)
        frames.append(frame)
        provenance.append({
            "url": url, "sha256": hashlib.sha256(data).hexdigest(), "bars": len(frame),
        })
    combined = pd.concat(frames).sort_index()
    if combined.index.has_duplicates:
        raise ValueError(f"{symbol}: duplicate candle timestamps")
    start = pd.Timestamp(WARMUP_START, tz="UTC")
    stop = pd.Timestamp(EVAL_END_EXCLUSIVE, tz="UTC")
    combined = combined.loc[(combined.index >= start) & (combined.index < stop)]
    expected = pd.date_range(
        start, stop - pd.Timedelta(timeframe), freq=timeframe, tz="UTC"
    )
    if not combined.index.equals(expected):
        missing = expected.difference(combined.index)
        extra = combined.index.difference(expected)
        raise ValueError(
            f"{symbol} ({timeframe}): incomplete candles, "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    if (
        not (combined[["open", "high", "low", "close"]] > 0).all().all()
        or (combined["volume"] < 0).any()
        or (combined["high"] < combined[["open", "close", "low"]].max(axis=1)).any()
        or (combined["low"] > combined[["open", "close", "high"]].min(axis=1)).any()
    ):
        raise ValueError(f"{symbol}: invalid OHLCV")
    return combined, provenance


class FrozenBars:
    name = "binance_vision_verified"

    def __init__(self, code: str, frame: pd.DataFrame) -> None:
        self.code, self.frame = code, frame

    def fetch(self, codes, start_date, end_date, *, interval="1H", fields=None):
        if list(codes) != [self.code]:
            raise ValueError("Single-instrument run required")
        return {self.code: self.frame}


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return safe_json(value.item())
    return value


def run_one(code: str, frame: pd.DataFrame, directory: Path, timeframe: str):
    config = {
        "codes": [code], "start_date": WARMUP_START,
        "end_date": "2026-11-11", "evaluation_start_date": EVAL_START,
        "source": "local", "interval": timeframe, "engine": "daily",
        "initial_cash": 1000.0, "leverage": 10.0,
        "margin_mode": "cross", "perpetual_strict": False,
        "position_adjustment": "rebalance", "rebalance_tolerance": 0.20,
        "taker_rate": 0.0005, "maker_rate": 0.0002,
        "slippage": 0.0003, "funding_rate": 0.0, "benchmark": None,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "artifacts").mkdir(exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    engine = CryptoEngine(config)
    metrics = engine.run_backtest(
        config, FrozenBars(code, frame), SignalEngine(), directory,
        bars_per_year=365 * (24 if timeframe == "1H" else 96),
    )
    (directory / "metrics_full.json").write_text(
        json.dumps(safe_json(metrics), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "symbol": code, "timeframe": timeframe,
        "start": EVAL_START, "end_exclusive": EVAL_END_EXCLUSIVE,
        "bars": int((frame.index >= pd.Timestamp(EVAL_START, tz="UTC")).sum()),
        "trades": len(engine.trades), "metrics": safe_json(metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", choices=("1H", "15m"), default="1H")
    parser.add_argument("--output", default="video-five-backtest-results")
    args = parser.parse_args()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    results = []
    with requests.Session() as session:
        for symbol in SYMBOLS:
            data, sources = historical_data(
                symbol, "1h" if args.timeframe == "1H" else "15m", session,
            )
            code = symbol.replace("USDT", "-USDT")
            directory = target / f"{symbol}_{args.timeframe}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "sources.json").write_text(
                json.dumps(sources, indent=2), encoding="utf-8"
            )
            row = run_one(code, data, directory, args.timeframe)
            results.append(row)
            print(
                code, args.timeframe, "trades:", row["trades"],
                "return:", row["metrics"].get("total_return"),
                "max DD:", row["metrics"].get("max_drawdown"),
                "win rate:", row["metrics"].get("win_rate"),
            )
    report = {
        "video": "https://youtu.be/d5tsDoX7U3w",
        "method": "Vibe CryptoEngine, confirmed HA signals, real next-open fills",
        "limitations": [
            "10x cross approximated, strict intrabar liquidation and mark prices not modeled",
            "Historical funding rates not included",
            "TP/SL candle contact is observed then closed on next real open",
            "Confirmation windows and stop buffer are research assumptions",
        ],
        "results": results,
    }
    (target / "summary.json").write_text(
        json.dumps(report, indent=2, default=str, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
