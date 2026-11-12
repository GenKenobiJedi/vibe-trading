"""Causal five-indicator strategy inspired by youtu.be/d5tsDoX7U3w.

Signals on CLOSED Heikin-Ashi candles, Vibe Trading fills on the next REAL open.
The video does not specify exact setup windows or stop buffer: these are
explicit backtest assumptions, not claimed to be original author settings.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def wilder(values: pd.Series, period: int) -> pd.Series:
    """TradingView-like RMA with SMA initialization (no look-ahead)."""
    source = values.astype(float)
    seed = source.rolling(period, min_periods=period).mean().to_numpy()
    raw = source.to_numpy()
    out = np.full(len(source), np.nan)
    ready = np.flatnonzero(np.isfinite(seed))
    if not len(ready):
        return pd.Series(out, index=source.index)
    start = int(ready[0])
    out[start] = seed[start]
    for i in range(start + 1, len(raw)):
        if np.isfinite(raw[i]):
            out[i] = (out[i - 1] * (period - 1) + raw[i]) / period
    return pd.Series(out, index=source.index)


def heikin_ashi(frame: pd.DataFrame) -> pd.DataFrame:
    """Use synthetic candles only for INDICATORS, never for execution."""
    o, h, l, c = (frame[col].astype(float).to_numpy() for col in ("open", "high", "low", "close"))
    hc = (o + h + l + c) / 4.0
    ho = np.empty_like(hc)
    ho[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(hc)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2.0
    return pd.DataFrame(
        {"open": ho, "high": np.maximum.reduce([h, ho, hc]),
         "low": np.minimum.reduce([l, ho, hc]), "close": hc},
        index=frame.index,
    )


def rsi_bands(close: pd.Series, period: int = 15) -> pd.DataFrame:
    """LazyBear RSI price-projection bands, with OB/OS 70/30."""
    up = close.diff().clip(lower=0).ewm(span=2 * period - 1, adjust=False).mean()
    down = (-close.diff()).clip(lower=0).ewm(span=2 * period - 1, adjust=False).mean()
    def level(target: float) -> pd.Series:
        x = (period - 1) * (down * target / (100 - target) - up)
        return close + x.where(x >= 0, x * (100 - target) / target)
    upper, lower = level(70.0), level(30.0)
    return pd.DataFrame({"lower": lower, "middle": (upper + lower) / 2, "upper": upper})


def stochastic_rsi(close: pd.Series, rsi_period: int = 8, stoch_period: int = 10) -> pd.Series:
    diff = close.diff()
    gain = wilder(diff.clip(lower=0), rsi_period)
    loss = wilder((-diff).clip(lower=0), rsi_period)
    denominator = gain + loss
    rsi = (100.0 * gain / denominator.replace(0, np.nan)).fillna(50.0)
    lo = rsi.rolling(stoch_period, min_periods=stoch_period).min()
    hi = rsi.rolling(stoch_period, min_periods=stoch_period).max()
    raw_k = 100 * (rsi - lo) / (hi - lo).replace(0, np.nan)
    return raw_k.rolling(3, min_periods=3).mean()  # K=3, D=3 hidden


def two_pole_points(close: pd.Series, length: int = 12) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Independent mathematical reconstruction of BigBeluga's two-pole dots.

    Critically, the original indicator draws a CONFIRMED dot one candle to
    the left. The returned event belongs to the actual confirmation candle.
    """
    base = close.rolling(25, min_periods=25).mean()
    deviation = close - base
    normalized = (deviation - deviation.rolling(25, min_periods=25).mean()) / (
        deviation.rolling(25, min_periods=25).std(ddof=0).replace(0, np.nan)
    )
    alpha = 2.0 / (length + 1)
    output = np.full(len(close), np.nan)
    first = second = np.nan
    for i, v in enumerate(normalized.to_numpy()):
        if not np.isfinite(v):
            continue
        first = v if not np.isfinite(first) else (1 - alpha) * first + alpha * v
        second = first if not np.isfinite(second) else (1 - alpha) * second + alpha * first
        output[i] = second
    oscillator = pd.Series(output, index=close.index)
    lag = oscillator.shift(4)
    buy = (oscillator > lag) & (oscillator.shift(1) <= lag.shift(1)) & (oscillator < 0)
    sell = (oscillator < lag) & (oscillator.shift(1) >= lag.shift(1)) & (oscillator > 0)
    return oscillator, buy.fillna(False), sell.fillna(False)


def supertrend(ha: pd.DataFrame, period: int = 10, multiplier: float = 1.2) -> pd.Series:
    """Supertrend variant with hl2 source and Wilder ATR (changeATR=True)."""
    close, high, low = (ha[x].astype(float) for x in ("close", "high", "low"))
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = wilder(tr, period).to_numpy()
    hl2 = ((high + low) / 2).to_numpy()
    c = close.to_numpy()
    direction = np.zeros(len(ha), dtype=int)
    up, dn = np.nan, np.nan
    trend = 1
    for i in range(len(ha)):
        if not np.isfinite(atr[i]):
            continue
        raw_up = hl2[i] - multiplier * atr[i]
        raw_dn = hl2[i] + multiplier * atr[i]
        prev_up, prev_dn = up, dn
        if i and np.isfinite(prev_up) and c[i - 1] > prev_up:
            up = max(raw_up, prev_up)
        else:
            up = raw_up
        if i and np.isfinite(prev_dn) and c[i - 1] < prev_dn:
            dn = min(raw_dn, prev_dn)
        else:
            dn = raw_dn
        if trend == -1 and np.isfinite(prev_dn) and c[i] > prev_dn:
            trend = 1
        elif trend == 1 and np.isfinite(prev_up) and c[i] < prev_up:
            trend = -1
        direction[i] = trend
    return pd.Series(direction, index=ha.index)


class SignalEngine:
    """Return margin target weights; Vibe handles 10x and next-open fills.

    Assumptions absent from the video: setup/confirmation windows, SL buffer
    0.20%, hard cap at 10% of initial margin, and position sizing 10% of equity.
    """
    def __init__(
        self, stake: float = 0.10, setup_bars: int = 24,
        confirmation_bars: int = 12, stop_buffer: float = 0.002,
        leverage: float = 10.0, margin_loss_cap: float = 0.10,
        stoch_up: float = 98.0, stoch_down: float = 2.0,
        supertrend_multiplier: float = 1.2,
    ) -> None:
        self.stake = stake
        self.setup_bars = setup_bars
        self.confirmation_bars = confirmation_bars
        self.stop_buffer = stop_buffer
        self.leverage = leverage
        self.margin_loss_cap = margin_loss_cap
        self.stoch_up = stoch_up
        self.stoch_down = stoch_down
        self.supertrend_multiplier = supertrend_multiplier

    def generate(self, data_map: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        return {symbol: self._one(frame) for symbol, frame in data_map.items()}

    def _one(self, frame: pd.DataFrame) -> pd.Series:
        if len(frame) < 250:
            return pd.Series(0.0, index=frame.index)
        ha = heikin_ashi(frame)
        close = ha["close"]
        bands = rsi_bands(close)
        _, buy, sell = two_pole_points(close)
        k = stochastic_rsi(close)
        st = supertrend(ha, multiplier=self.supertrend_multiplier)
        ema200 = close.ewm(span=200, adjust=False, min_periods=200).mean()
        low = frame["low"].to_numpy(dtype=float)
        high = frame["high"].to_numpy(dtype=float)
        market = frame["close"].to_numpy(dtype=float)
        ha_low, ha_high = ha["low"].to_numpy(), ha["high"].to_numpy()
        lower, middle, upper = (bands[x].to_numpy() for x in ("lower", "middle", "upper"))
        k_vals, st_vals, ema_vals = k.to_numpy(), st.to_numpy(), ema200.to_numpy()
        buy_vals, sell_vals = buy.to_numpy(), sell.to_numpy()
        signals = np.zeros(len(frame), dtype=float)
        low_touch = high_touch = last_buy = last_sell = -10_000
        up_flip = down_flip = k_high = k_low = -10_000
        used_buy = used_sell = -10_000
        side, half, entry_price, stop_price = 0, False, 0.0, 0.0
        for i in range(len(frame)):
            if not np.isfinite(ema_vals[i]) or not np.isfinite(lower[i]):
                continue
            if ha_low[i] <= lower[i]:
                low_touch = i
            if ha_high[i] >= upper[i]:
                high_touch = i
            if buy_vals[i]:
                last_buy = i
            if sell_vals[i]:
                last_sell = i
            if i and st_vals[i] == 1 and st_vals[i - 1] == -1:
                up_flip = i
            if i and st_vals[i] == -1 and st_vals[i - 1] == 1:
                down_flip = i
            if np.isfinite(k_vals[i]) and k_vals[i] >= self.stoch_up:
                k_high = i
            if np.isfinite(k_vals[i]) and k_vals[i] <= self.stoch_down:
                k_low = i
            # Exit decisions use fully observed current candle; execution
            # happens at NEXT real open inside Vibe (never at band price).
            if side:
                stop_hit = low[i] <= stop_price if side == 1 else high[i] >= stop_price
                tp2_hit = high[i] >= upper[i] if side == 1 else low[i] <= lower[i]
                tp1_hit = high[i] >= middle[i] if side == 1 else low[i] <= middle[i]
                if stop_hit or tp2_hit:
                    side, half = 0, False
                elif tp1_hit:
                    half = True
            if not side:
                go_long = (
                    low_touch <= last_buy and i - low_touch <= self.setup_bars
                    and last_buy > used_buy and i - last_buy <= self.confirmation_bars
                    and i - up_flip <= self.confirmation_bars
                    and i - k_high <= self.confirmation_bars
                    and st_vals[i] == 1 and close.iat[i] > ema_vals[i]
                    and market[i] < middle[i] < upper[i]
                )
                go_short = (
                    high_touch <= last_sell and i - high_touch <= self.setup_bars
                    and last_sell > used_sell and i - last_sell <= self.confirmation_bars
                    and i - down_flip <= self.confirmation_bars
                    and i - k_low <= self.confirmation_bars
                    and st_vals[i] == -1 and close.iat[i] < ema_vals[i]
                    and market[i] > middle[i] > lower[i]
                )
                if go_long:
                    side, half, used_buy = 1, False, last_buy
                    entry_price = market[i]
                    band_stop = lower[i] * (1.0 - self.stop_buffer)
                    cap_stop = entry_price * (1.0 - self.margin_loss_cap / self.leverage)
                    # A band above the entry is not a valid LONG stop.
                    stop_price = max(band_stop, cap_stop) if 0 < band_stop < entry_price else cap_stop
                elif go_short:
                    side, half, used_sell = -1, False, last_sell
                    entry_price = market[i]
                    band_stop = upper[i] * (1.0 + self.stop_buffer)
                    cap_stop = entry_price * (1.0 + self.margin_loss_cap / self.leverage)
                    # A band below the entry is not a valid SHORT stop.
                    stop_price = min(band_stop, cap_stop) if band_stop > entry_price else cap_stop
            signals[i] = side * self.stake * (0.5 if half else 1.0)
        return pd.Series(signals, index=frame.index, dtype=float)
