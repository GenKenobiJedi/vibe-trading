"""Offline regression tests for the YouTube indicator strategy."""
import numpy as np
import pandas as pd

from experiments.video_five_indicators.signal_engine import (
    SignalEngine, heikin_ashi, rsi_bands, stochastic_rsi, two_pole_points,
)
from experiments.video_five_indicators.run import archive_urls, read_archive
import io
import zipfile


def prices(count=320, ascending=True):
    t = pd.date_range("2026-07-01", periods=count, freq="h", tz="UTC")
    base = np.linspace(100, 150 if ascending else 50, count)
    wiggle = 3 * np.sin(np.arange(count) / 6)
    close = base + wiggle
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame({
        "open": op, "high": np.maximum(op, close) + 0.1,
        "low": np.minimum(op, close) - 0.1,
        "close": close, "volume": 100.0,
    }, index=t)


def test_heikin_ashi_has_no_future_dependency():
    frame = prices()
    short = heikin_ashi(frame.iloc[:200])
    whole = heikin_ashi(frame)
    pd.testing.assert_frame_equal(short, whole.iloc[:200])


def test_two_pole_dots_are_confirmed_bar_events():
    frame = prices()
    _, buy_short, sell_short = two_pole_points(frame.close.iloc[:200])
    _, buy_full, sell_full = two_pole_points(frame.close)
    pd.testing.assert_series_equal(buy_short, buy_full.iloc[:200])
    pd.testing.assert_series_equal(sell_short, sell_full.iloc[:200])


def test_no_lookahead_on_signal_prefix():
    frame = prices(380)
    result = SignalEngine().generate({"BTC-USDT": frame})["BTC-USDT"]
    truncated = SignalEngine().generate({"BTC-USDT": frame.iloc[:300]})["BTC-USDT"]
    pd.testing.assert_series_equal(truncated, result.iloc[:300])


def test_neutral_short_history():
    frame = prices(100)
    result = SignalEngine().generate({"BTC-USDT": frame})["BTC-USDT"]
    assert (result == 0).all()


def test_valid_target_and_zero_fee_simulation_inputs():
    frame = prices(400)
    values = SignalEngine().generate({"BTC-USDT": frame})["BTC-USDT"]
    assert values.index.equals(frame.index)
    assert values.notna().all()
    assert values.abs().max() <= 0.1


def test_rsi_bands_order_after_warmup():
    band = rsi_bands(prices().close).iloc[100:]
    assert (band.lower < band.middle).all()
    assert (band.middle < band.upper).all()


def test_stoch_rsi_is_bounded():
    k = stochastic_rsi(prices().close).dropna()
    assert not k.empty
    assert k.between(-1e-8, 100 + 1e-8).all()


def test_archive_has_correct_coverage():
    urls = archive_urls("BTCUSDT", "1h")
    assert len(urls) == 15
    assert urls[0].endswith("2026-07.zip")
    assert urls[-1].endswith("2026-11-11.zip")


def test_microsecond_archive_parsed_without_header():
    row = "1794355200000000,100,102,99,101,1000,1794358799999999,0,0,0,0,0\n"
    handle = io.BytesIO()
    with zipfile.ZipFile(handle, "w") as zip_file:
        zip_file.writestr("one.csv", row)
    frame = read_archive(handle.getvalue())
    assert frame.index[0].year == 2026
    assert frame.open.iat[0] == 100
