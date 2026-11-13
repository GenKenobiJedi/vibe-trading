"""Point-in-time research features, independently derived from source concepts.

Each input row describes a completed minute, with its UNIX OPEN timestamp.
HTF observations are joined on close time. No runtime services are imported.
"""
from __future__ import annotations

import hashlib
import inspect
from numbers import Real
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import talib

FIELDS = ['time', 'open', 'high', 'low', 'close', 'volume']
SIDES = [('long', 1), ('short', -1)]


def _validate(df: pd.DataFrame, seconds: int) -> None:
    if (not isinstance(df, pd.DataFrame) or df.empty or
            not isinstance(df.index, pd.RangeIndex) or df.index.step <= 0 or
            not df.columns.is_unique or not set(FIELDS).issubset(df.columns)):
        raise ValueError('Nonempty OHLCV DataFrame with increasing RangeIndex required')
    t = df.time.to_numpy()
    if (t.dtype.kind not in 'iu' or np.any(t < 0) or np.any(t > 253402286399) or
            np.any(t % seconds) or np.any(np.diff(t.astype(np.int64)) != seconds)):
        raise ValueError('Integer UNIX open seconds must be aligned, bounded and contiguous')
    if any(df[k].dtype.kind not in 'iuf' for k in FIELDS[1:]):
        raise ValueError('Numeric OHLCV required')
    a = df[FIELDS[1:]].to_numpy(dtype=float)
    o, h, low, c, v = a.T
    if (not np.isfinite(a).all() or np.any(a[:, :4] <= 0) or np.any(v < 0) or
            np.any(low > np.minimum(o, c)) or np.any(h < np.maximum(o, c))):
        raise ValueError('Invalid finite positive OHLC prices or nonnegative volume')


def _aggregate(df: pd.DataFrame, seconds: int) -> pd.DataFrame:
    key = df.time // seconds * seconds
    grouped = df.groupby(key, sort=True)
    out = grouped.agg(open=('open', 'first'), high=('high', 'max'), low=('low', 'min'),
                      close=('close', 'last'), volume=('volume', 'sum'))
    out = out.loc[grouped.size() == seconds // 60].reset_index()
    return out[FIELDS]


def _align(closes: np.ndarray, known: np.ndarray, values: np.ndarray) -> np.ndarray:
    position = np.searchsorted(known, closes, side='right') - 1
    out = np.full(len(closes), np.nan)
    valid = position >= 0
    out[valid] = np.asarray(values)[position[valid]]
    return out


def _trend(df: pd.DataFrame, fast: int, slow: int, warm: int, *, price_slope: bool = True) -> tuple:
    c = df.close.to_numpy(dtype=float)
    e1, e2 = talib.EMA(c, fast), talib.EMA(c, slow)
    slope = pd.Series(e1).diff(3).to_numpy()
    ready = np.arange(len(df)) >= warm - 1
    bullish = (c > e1) & (slope > 0) if price_slope else True
    bearish = (c < e1) & (slope < 0) if price_slope else True
    direction = np.select([ready & (e1 > e2) & bullish,
                           ready & (e1 < e2) & bearish], [1, -1], 0).astype(np.int8)
    return direction, e1, e2, slope


def _load_scanner(path: Path) -> tuple:
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    module = ModuleType('_research_ict_explicit')
    module.__file__ = str(path)
    # Execute the explicitly authorized local module without writing bytecode/cache.
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    scanner = getattr(module, 'ict_scan', None)
    if not callable(scanner) or 'timestamps' not in inspect.signature(scanner).parameters:
        raise ValueError('Corrected ict_scan with explicit timestamps parameter required')
    return scanner, {'ict_module_path': str(path), 'ict_module_sha256': hashlib.sha256(raw).hexdigest()}


def _higher(df: pd.DataFrame, seconds: int, prelude: pd.DataFrame | None) -> pd.DataFrame:
    derived = _aggregate(df, seconds)
    if prelude is None:
        return derived
    _validate(prelude, seconds)
    if np.any(prelude.time.to_numpy() + seconds > int(df.time.iloc[0])):
        raise ValueError('HTF warmup must close before minute input begins')
    joined = pd.concat([prelude[FIELDS], derived], ignore_index=True)
    _validate(joined, seconds)
    return joined


def _scan_context(htf: pd.DataFrame, scanner, seconds: int) -> tuple:
    scores = np.full((len(htf), 2), np.nan)
    contexts = [[[], []] for _ in range(len(htf))]
    prices = [htf[k].to_numpy(dtype=float) for k in FIELDS[1:5]]
    times = htf.time.to_numpy(dtype=np.int64)
    calls = 0
    for end in range(199, len(htf)):
        start = end - 199
        for column, (side, _) in enumerate(SIDES):
            result = scanner(*(a[start:end + 1].copy() for a in prices),
                             direction=side.upper(), require_killzone=False,
                             timestamps=times[start:end + 1].copy() + seconds)
            calls += 1
            score = result['score']
            if isinstance(score, bool) or not np.isfinite(score) or score < 0:
                raise ValueError('Invalid ICT score')
            scores[end, column] = score
            details = result['details']
            for kind, zones in [('fvg', details['fvgs']), ('ob', details['obs'])]:
                for z in zones:
                    if (not z.get('active', False) or not z.get('confirmed', False) or
                            not z['type'].startswith('bullish' if column == 0 else 'bearish')):
                        continue
                    # Core details retain diagnostic previews, including stale
                    # untouched zones. Apply the same candle-index age policy.
                    formed = z.get('formed_at', z.get('index'))
                    if (not isinstance(formed, (int, np.integer)) or
                            isinstance(formed, (bool, np.bool_)) or not 180 <= formed <= 199):
                        continue
                    top, bottom = ((z['gap_top'], z['gap_bottom']) if kind == 'fvg'
                                   else (z['ob_high'], z['ob_low']))
                    known = z['known_at']
                    if not isinstance(known, (int, np.integer)) or not 0 <= known <= 199:
                        raise ValueError('Invalid ICT zone known_at')
                    if not np.isfinite([top, bottom]).all() or not 0 < bottom < top:
                        raise ValueError('Invalid ICT zone bounds')
                    contexts[end][column].append((float(bottom), float(top), int(times[start + known]), kind))
    return scores, contexts, calls


def _pivots(df: pd.DataFrame, window: int) -> tuple:
    """Emit unique swing levels only on their confirmation candle."""
    result = []
    for name, high_side in [('high', True), ('low', False)]:
        a = df[name].to_numpy(dtype=float)
        event = np.full(len(a), np.nan)
        width = 2 * window + 1
        if len(a) >= width:
            view = np.lib.stride_tricks.sliding_window_view(a, width)
            extreme = view.max(axis=1) if high_side else view.min(axis=1)
            valid = ((view[:, window] == extreme) &
                     ((view == extreme[:, None]).sum(axis=1) == 1))
            event[width - 1:] = np.where(valid, extreme, np.nan)
        result.append(event)
    return tuple(result)


def _formations(df: pd.DataFrame) -> dict:
    """Local 1m FVG or two-candle displacement OB; known only at close."""
    o, h, low, c = (df[k].to_numpy(dtype=float) for k in FIELDS[1:5])
    zones = {}
    for i in range(1, len(df)):
        new = []
        if i >= 2 and low[i] > h[i - 2]:
            new.append((1, h[i - 2], low[i], i, i, 'fvg'))
        if i >= 2 and h[i] < low[i - 2]:
            new.append((-1, h[i], low[i - 2], i, i, 'fvg'))
        if c[i - 1] < o[i - 1] and c[i] > h[i - 1] and c[i] > o[i]:
            new.append((1, low[i - 1], h[i - 1], i, i - 1, 'ob'))
        if c[i - 1] > o[i - 1] and c[i] < low[i - 1] and c[i] < o[i]:
            new.append((-1, low[i - 1], h[i - 1], i, i - 1, 'ob'))
        if new:
            zones[i] = new
    return zones


def _sequence(df: pd.DataFrame, directions: np.ndarray) -> dict:
    """One-pass sweep, subsequent close break, OTE, then fresh first tap."""
    n = len(df)
    h, low, c = (df[k].to_numpy(dtype=float) for k in ['high', 'low', 'close'])
    hp, lp = _pivots(df, 3)
    formations = _formations(df)
    output = {}
    for side, sign in SIDES:
        for name in ['sweep', 'structure', 'ote', 'fresh_zone']:
            output[name + '_' + side] = np.zeros(n, dtype=bool)
        for name in ['ote_lower', 'ote_upper', 'structural_stop', 'sweep_at', 'structure_at', 'ote_at']:
            output[name + '_' + side] = np.full(n, np.nan)
        last_high = last_low = np.nan
        setup = None
        active = []
        for i in range(n):
            sign_valid = directions[i] == sign
            if setup and (not sign_valid or i - setup['sweep'] > 60 or
                          (low[i] < setup['low'] if sign == 1 else h[i] > setup['high'])):
                setup = None
            # Old confirmed levels only: today's confirmation is applied below.
            swept = (low[i] < last_low < c[i] if sign == 1 else h[i] > last_high > c[i])
            if setup is None and sign_valid and swept and np.isfinite(last_high + last_low):
                level = last_high if sign == 1 else last_low
                if sign * (c[i] - level) <= 0:
                    setup = dict(sweep=i, structure=None, ote=None, level=level, low=low[i], high=h[i])
            if setup:
                if setup['structure'] is None:
                    setup['low'] = min(setup['low'], low[i])
                    setup['high'] = max(setup['high'], h[i])
                    if (i > setup['sweep'] and sign * (c[i] - setup['level']) > 0 and
                            sign * (c[i - 1] - setup['level']) <= 0):
                        setup['structure'] = i
                        span = setup['high'] - setup['low']
                        setup['lower'] = setup['high'] - .79 * span if sign == 1 else setup['low'] + .62 * span
                        setup['upper'] = setup['high'] - .62 * span if sign == 1 else setup['low'] + .79 * span
                elif (setup['ote'] is None and i > setup['structure'] and
                      setup['lower'] <= c[i] <= setup['upper']):
                    setup['ote'] = i
                output['sweep_' + side][i] = True
                output['structure_' + side][i] = setup['structure'] is not None
                output['ote_' + side][i] = setup['ote'] is not None
                for name in ['sweep', 'structure', 'ote']:
                    if setup[name] is not None:
                        output[name + '_at_' + side][i] = setup[name]
                if setup['structure'] is not None:
                    output['ote_lower_' + side][i], output['ote_upper_' + side][i] = setup['lower'], setup['upper']
                    output['structural_stop_' + side][i] = setup['low'] if sign == 1 else setup['high']
            survivors = []
            for z in active:
                _, bottom, top, known, origin, _ = z
                crossed = low[i] < bottom if sign == 1 else h[i] > top
                touched = low[i] <= top and h[i] >= bottom
                if i - origin > 30 or crossed:
                    continue
                if touched:
                    if setup and setup['ote'] is not None and i > setup['ote'] and known < i:
                        overlap_low = max(bottom, setup['lower'])
                        overlap_high = min(top, setup['upper'])
                        output['fresh_zone_' + side][i] |= (
                            overlap_low <= overlap_high and
                            low[i] <= overlap_high and h[i] >= overlap_low and
                            setup['lower'] <= c[i] <= setup['upper'])
                    # Any first tap retires a zone, including failed reactions.
                    continue
                survivors.append(z)
            active = survivors + [z for z in formations.get(i, []) if z[0] == sign]
            if output['fresh_zone_' + side][i]:
                setup = None
            if np.isfinite(hp[i]):
                last_high = hp[i]
            if np.isfinite(lp[i]):
                last_low = lp[i]
    return output


def _baseline(frame: pd.DataFrame, contexts: dict) -> dict:
    """Align active direction-matching HTF zones and retire on minute touches."""
    n = len(frame)
    times = frame.time.to_numpy(dtype=np.int64) + 60
    high, low, close = (frame[k].to_numpy(dtype=float) for k in ['high', 'low', 'close'])
    result = {}
    for column, (side, sign) in enumerate(SIDES):
        near = np.zeros(n, dtype=bool)
        for key, seconds in [('1h', 3600), ('4h', 14400)]:
            known, snapshots = contexts[key]
            retired = set()
            first = max(0, int(np.searchsorted(known, times[0], side='right')) - 1)
            for j in range(first, len(known)):
                start = int(np.searchsorted(times, known[j], side='left'))
                end = int(np.searchsorted(times, known[j + 1], side='left')) if j + 1 < len(known) else n
                if start >= end:
                    continue
                for bottom, top, origin, kind in snapshots[j][column]:
                    identity = (origin, kind, bottom, top)
                    if identity in retired:
                        continue
                    begin = max(start, int(np.searchsorted(times, origin + seconds, side='right')))
                    if begin >= end:
                        continue
                    touched = (low[begin:end] <= top) & (high[begin:end] >= bottom)
                    crossed = low[begin:end] < bottom if sign == 1 else high[begin:end] > top
                    retire = np.flatnonzero(touched | crossed)
                    stop = begin + int(retire[0]) + 1 if len(retire) else end
                    midpoint = (top + bottom) / 2
                    near[begin:stop] |= ((np.abs(close[begin:stop] - midpoint) / close[begin:stop] < .02) &
                                         ~(low[begin:stop] < bottom if sign == 1 else high[begin:stop] > top))
                    if len(retire):
                        retired.add(identity)
        result['ict_zone_' + side] = near
        result['baseline_' + side] = (near & frame.ready.to_numpy() &
            (frame.trend5.to_numpy() == sign) & (frame.trend1h.to_numpy() == sign) &
            (frame.trend4h.to_numpy() == sign) & (frame['ict_score1h_' + side].to_numpy() >= 6) &
            (frame['ict_score4h_' + side].to_numpy() >= 5) & frame['reaction_' + side].to_numpy() &
            (frame.volume.to_numpy() > 0))
    return result


def _minute_indicators(df: pd.DataFrame) -> dict:
    o, high, low, close, volume = (df[k].to_numpy(dtype=float) for k in FIELDS[1:])
    hc = (o + high + low + close) / 4
    ho = np.empty(len(df))
    ho[0] = (o[0] + close[0]) / 2
    for i in range(1, len(df)):
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    hh, hl = np.maximum.reduce([high, ho, hc]), np.minimum.reduce([low, ho, hc])
    typical = pd.Series((hh + hl + hc) / 3)
    mean, std = typical.rolling(40).mean(), typical.rolling(40).std(ddof=1)
    lower, upper = mean - 2 * std, mean + 2 * std
    rsi = pd.Series(talib.RSI(close, 14))
    bbmean = pd.Series(close).rolling(20).mean()
    width = 4 * pd.Series(close).rolling(20).std(ddof=1) / bbmean
    prior_volume = pd.Series(volume).rolling(24).mean().shift(12)
    volume12 = pd.Series(volume).rolling(12).mean()
    spread = pd.Series(talib.EMA(close, 9) - talib.EMA(close, 21))
    result = dict(ha_open=ho, ha_close=hc, ha_high=hh, ha_low=hl,
                  bb40_lower=lower.to_numpy(), bb40_upper=upper.to_numpy(), rsi14=rsi.to_numpy(),
                  bb20_width=width.to_numpy(), momentum_delta=spread.diff().to_numpy(),
                  volume_ratio12_24=(volume12 / prior_volume.where(prior_volume > 0)).to_numpy())
    for side, sign in SIDES:
        touch = pd.Series(hl <= lower if sign == 1 else hh >= upper)
        recent = touch.rolling(3, min_periods=3).max().eq(1)
        rsi_ok = rsi.between(30, 55) & (rsi.diff() > 0) if sign == 1 else rsi.between(45, 70) & (rsi.diff() < 0)
        result['pullback_' + side] = (recent & rsi_ok & (sign * (hc - ho) > 0) &
                                     (sign * np.diff(hc, prepend=np.nan) > 0)).to_numpy()
    return result


def _deadfish(frame: pd.DataFrame) -> dict:
    width = frame.bb20_width
    common = ((frame.volume_ratio12_24 < .8) & (width < width.shift(4)) &
              (width < .8 * width.rolling(50).median().shift(1)))
    return {'deadfish_' + side: (common & (sign * frame.momentum_delta <= 0).rolling(3).sum().eq(3) &
                                ~frame['strong_trend_' + side]).to_numpy()
            for side, sign in SIDES}


def build_features(df: pd.DataFrame, *, ict_module_path: Path,
                   htf_warmup: dict | None = None) -> pd.DataFrame:
    """Build causal features without changing input or writing external state.

    Args:
        df: Contiguous completed minute OHLCV with UNIX open seconds.
        ict_module_path: Explicit, trusted corrected local ict_analysis.py.
        htf_warmup: Optional actual prior closed 1h/4h OHLCV frames.

    Returns:
        A new frame with original columns, features and provenance attributes.

    Raises:
        ValueError: Malformed input, prelude or scanner contract.
    """
    _validate(df, 60)
    if htf_warmup is not None and (not isinstance(htf_warmup, dict) or
                                   not set(htf_warmup).issubset({'1h', '4h'})):
        raise ValueError('htf_warmup accepts only 1h and 4h')
    if htf_warmup is not None and any(not isinstance(value, pd.DataFrame) for value in htf_warmup.values()):
        raise ValueError('Each supplied HTF warmup must be an OHLCV DataFrame')
    scanner, provenance = _load_scanner(ict_module_path)
    out = df.copy(deep=True)
    minute_closes = df.time.to_numpy(dtype=np.int64) + 60
    counts = {'ict_scan_calls': 0}
    contexts = {}
    for key, seconds in [('5', 300), ('1h', 3600), ('4h', 14400)]:
        prelude = (htf_warmup or {}).get(key)
        htf = _higher(df, seconds, prelude)
        known = htf.time.to_numpy(dtype=np.int64) + seconds
        trend, _, _, slope = _trend(htf, 9 if key == '5' else 20, 21 if key == '5' else 50, 200,
                                         price_slope=key != '5')
        out['trend' + key] = np.nan_to_num(_align(minute_closes, known, trend)).astype(np.int8)
        counts['closed_bars_' + key] = len(htf)
        if key == '5':
            five = htf
            high, low, close = (htf[k].to_numpy(dtype=float) for k in ['high', 'low', 'close'])
            for name, values in [('atr', talib.ATR(high, low, close, 14)),
                                 ('adx5', talib.ADX(high, low, close, 14)), ('slope5', slope)]:
                out[name] = _align(minute_closes, known, values)
        else:
            scores, zones, calls = _scan_context(htf, scanner, seconds)
            counts['ict_scan_calls'] += calls
            contexts[key] = (known, zones)
            for column, (side, _) in enumerate(SIDES):
                out['ict_score' + key + '_' + side] = _align(minute_closes, known, scores[:, column])
    o, high, low, close = (df[k].to_numpy(dtype=float) for k in FIELDS[1:5])
    out['atr1m'] = talib.ATR(high, low, close, 14)
    out['ema9'], out['ema21'] = talib.EMA(close, 9), talib.EMA(close, 21)
    out['momentum'] = np.sign(out.ema9 - out.ema21).fillna(0).astype(np.int8)
    out['volume_prior20'] = df.volume.rolling(20).mean().shift(1)
    for side, sign in SIDES:
        out['reaction_' + side] = (sign * (close - o) > 0) & (sign * np.diff(close, prepend=np.nan) > 0)
        out['strong_trend_' + side] = (out.adx5 >= 25) & (out.trend5 == sign) & (sign * out.slope5 > 0)
        for prefix in ['baseline', 'sweep', 'ote', 'fresh_zone', 'pullback', 'deadfish']:
            out[prefix + '_' + side] = False
    out['ready'] = ((np.arange(len(out)) >= 599) & (out.trend5 != 0) &
                    (out.trend1h != 0) & (out.trend4h != 0))
    common = np.where(out.ready & (out.trend5 == out.trend1h) & (out.trend5 == out.trend4h), out.trend5, 0)
    for additions in [_minute_indicators(df), _sequence(df, common), _baseline(out, contexts)]:
        for name, values in additions.items():
            out[name] = values
    for name, values in _deadfish(out).items():
        out[name] = values
    pivoth, pivotl = _pivots(five, 2)
    known5 = five.time.to_numpy(dtype=np.int64) + 300
    for name, values in [('pivot5_high', pivoth), ('pivot5_low', pivotl)]:
        out[name] = _align(minute_closes, known5, pd.Series(values).ffill().to_numpy())
    counts.update({name: int(out[name].sum()) for name in out if out[name].dtype == bool})
    provenance.update(numpy_version=np.__version__, pandas_version=pd.__version__, talib_version=talib.__version__)
    counts['common_long'] = int((common == 1).sum())
    counts['common_short'] = int((common == -1).sum())
    out.attrs = {'definition': {
        'version': 'ict-freqle causal feature contract v1',
        'time': 'UNIX open seconds; completed HTF visible at inclusive close boundary',
        'scanner_timestamps': 'HTF evaluation CLOSE seconds',
        'warmup': {'1m': 600, '5m': 200, '1h': 200, '4h': 200},
        'ict_prefix_max': 200, 'ict_score_min': {'1h': 6, '4h': 5},
        'pivot_window': {'1m': 3, '5m': 2},
        'ote_retracement': [.62, .79], 'setup_expiry_minutes': 60,
        'local_zone_max_age_1m': 30, 'zone_retirement': 'first touch; no far-boundary crossing on entry',
        'ote_entry': 'fresh zone overlaps frozen OTE, tap intersects overlap, reaction close remains inside OTE',
        'volume_ratio_default': 1.2, 'volume_ratio_bounds': [1., 3.],
        'pullback': 'recursive HA; HA typical-price BB40 sample std; last3 completed bars; real-close RSI14',
        'deadfish': 'current volume12 / preceding nonoverlapping volume24; joint width/momentum weakness',
        'adaptation': 'independent concepts; no native Freqle, production ML or spot-to-futures parity claim',
    }, 'provenance': provenance, 'counts': counts}
    return out


def signals(frame: pd.DataFrame, variant: str, params: dict | None = None) -> np.ndarray:
    """Return closed-minute research opportunities; execution owns costs/fills.

    Args:
        frame: Output of build_features, in chronological order.
        variant: baseline, sweep_ote, pullback, or sweep_ote_pullback.
        params: Only volume_ratio is supported, with default 1.2 and bounds [1,3].

    Returns:
        An independent int8 array of short (-1), flat (0), and long (+1) signals.

    Raises:
        ValueError: Unknown variant or malformed/out-of-range parameters.
    """
    if variant not in {'baseline', 'sweep_ote', 'pullback', 'sweep_ote_pullback'}:
        raise ValueError('Unknown variant')
    params = {} if params is None else params
    if not isinstance(params, dict) or not set(params).issubset({'volume_ratio'}):
        raise ValueError('Only volume_ratio parameter supported')
    ratio = params.get('volume_ratio', 1.2)
    if isinstance(ratio, (bool, np.bool_)) or not isinstance(ratio, Real) or not np.isfinite(ratio) or not 1 <= ratio <= 3:
        raise ValueError('volume_ratio must be finite within [1, 3]')
    result = np.zeros(len(frame), dtype=np.int8)
    for side, sign in SIDES:
        gate = (frame.ready & (frame.trend5 == sign) & (frame.trend1h == sign) &
                (frame.trend4h == sign) & frame['reaction_' + side] & (frame.volume > 0))
        if variant.startswith('sweep_ote'):
            gate &= (frame['sweep_' + side] & frame['ote_' + side] & frame['fresh_zone_' + side] &
                     (frame.volume_prior20 > 0) & (frame.volume >= ratio * frame.volume_prior20))
        else:
            gate &= frame['baseline_' + side]
        if 'pullback' in variant:
            gate &= frame['pullback_' + side]
        result[gate.to_numpy()] = sign
    return result
