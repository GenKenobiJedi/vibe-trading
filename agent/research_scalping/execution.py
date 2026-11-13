"""Deterministic, offline, one-position linear futures research execution.

OHLC trade prices are proxy mid prices; the spread is ASSUMED. Entry/exit fills
include adverse half-spread and slippage. Cost diagnostics are already included
in gross PnL and must not be subtracted a second time. Quantity is risk based;
leverage changes margin, original ROI barriers, and the defensive liquidation
guard, never the linear PnL multiplier.

Signals are supplied by the caller, not recomputed here. A signal timestamp is
its candle CLOSE. Evaluation starts flat, and a pre-window candle cannot enter
at the first evaluation open. Only complete candles within the window execute.
Intrabar fills are timestamped at candle close because OHLC has no tick times.
Funding before that close uses pre-intrabar-fill quantity; funding at the close
is processed at the next open using remaining quantity. A settlement exactly at
an open exit is excluded; one exactly at entry is included. This ordering is an
explicit OHLC-resolution assumption, not evidence of historical tick ordering.

Extrema conservatively allow favorable-then-adverse ordering while held, bounded
by executable barriers. On a profit-exit bar the adverse mark precedes the
profit fill; later prices cannot mark an exited position. Partial fills reduce
the quantity of subsequent marks. Extrema are observations, not realized fills.
The isolated liquidation guard assumes constant maintenance on
mark notional, excludes exchange tiers/fees/insurance, and is NOT an exchange
liquidation model. No funding argument means no supplied settlements; callers
must disclose missing funding evidence instead of claiming known-zero funding.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostConfig:
    fee: float = .0005
    slippage: float = .0002
    spread_bps: float = 2.
    leverage: float = 10.
    risk_usdt: float = 10.
    initial_equity: float = 1000.
    max_hold_bars: int = 15
    cooldown_bars: int = 2
    stop_atr: float = 1.2
    target_atr: float = 1.8
    exit_mode: str = 'atr'
    maintenance_fraction: float = .005
    use_source_exits: bool = True
    trailing_atr: float | None = None
    min_net_rr: float | None = None
    deadfish_enabled: bool = False
    deadfish_min_hold_bars: int = 5
    deadfish_stall_risk_fraction: float = .25

    def __post_init__(self):
        if not isinstance(self.use_source_exits, bool):
            raise ValueError('use_source_exits must be boolean')
        if not isinstance(self.deadfish_enabled, bool):
            raise ValueError('deadfish_enabled must be boolean')
        for name, minimum in (('trailing_atr', 0.), ('min_net_rr', 0.),
                              ('deadfish_stall_risk_fraction', 0.)):
            value = getattr(self, name)
            if value is None and name != 'deadfish_stall_risk_fraction':
                continue
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or
                    not isfinite(value) or value < minimum or (name == 'trailing_atr' and value == 0)):
                raise ValueError(f'{name} must be finite and within its positive/nonnegative bounds')
        if self.deadfish_stall_risk_fraction > 1:
            raise ValueError('deadfish_stall_risk_fraction must be <= 1')
        for name in ('fee', 'slippage', 'spread_bps', 'leverage', 'risk_usdt',
                     'initial_equity', 'stop_atr', 'target_atr', 'maintenance_fraction'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f'{name} must be finite and numeric')
        for name in ('fee', 'slippage', 'spread_bps', 'maintenance_fraction'):
            if getattr(self, name) < 0:
                raise ValueError(f'{name} must be nonnegative')
        for name in ('leverage', 'risk_usdt', 'initial_equity', 'stop_atr', 'target_atr'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.leverage < 1 or self.maintenance_fraction >= 1 / self.leverage:
            raise ValueError('leverage must be >=1 and initial margin must exceed maintenance')
        if self.fee >= 1 or self.slippage + self.spread_bps / 20000 >= 1:
            raise ValueError('cost fractions must leave positive execution prices')
        for name, minimum in (('max_hold_bars', 1), ('cooldown_bars', 0), ('deadfish_min_hold_bars', 1)):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if not isinstance(self.exit_mode, str) or self.exit_mode not in {'original', 'atr', 'partial', 'trailing', 'momentum'}:
            raise ValueError('unknown exit_mode')


def net_reward_risk(mid: float, side: int, stop: float, target: float, config: CostConfig) -> float:
    """Net target reward divided by the engine's conservative stop-plus-cost risk.

    Prices are fixed mid-price barriers. Invalid plans return negative infinity
    and are never repaired or replaced by a different opportunity.
    """
    impact = config.slippage + config.spread_bps / 20000
    entry = mid * (1 + side * impact)
    if (side not in (-1, 1) or not all(isfinite(v) and v > 0 for v in (mid, stop, target)) or
            side * (entry - stop) <= 0 or side * (target - entry) <= 0):
        return -float('inf')
    stop_fill, target_fill = stop * (1 - side * impact), target * (1 - side * impact)
    risk = max(abs(entry - stop) + 2 * entry * (config.fee + impact),
               side * (entry - stop_fill) + config.fee * (entry + stop_fill))
    return (side * (target_fill - entry) - config.fee * (entry + target_fill)) / risk


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer UNIX second')
    return int(value)


def _numeric(values, name, size, *, nan=False):
    values = np.asarray(values)
    if values.shape != (size,) or values.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must be a numeric one-dimensional array of length {size}')
    result = values.astype(float, copy=False)
    if np.isinf(result).any() or (not nan and np.isnan(result).any()):
        raise ValueError(f'{name} contains non-finite values')
    return result


def _inputs(df, signal, config, start, end, stops, targets, funding):
    if not isinstance(config, CostConfig):
        raise ValueError('config must be CostConfig')
    start, end = _integer(start, 'start'), _integer(end, 'end')
    if end <= start:
        raise ValueError('end must be greater than start')
    if not isinstance(df, pd.DataFrame) or not isinstance(df.index, pd.RangeIndex):
        raise ValueError('df must have an increasing RangeIndex')
    if df.index.step <= 0 or not df.columns.is_unique:
        raise ValueError('df must have unique columns and an increasing RangeIndex')
    required = {'time', 'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(df.columns):
        raise ValueError(f'missing columns: {sorted(required - set(df.columns))}')
    size = len(df)
    times = df['time'].to_numpy()
    if (times.dtype.kind not in 'iu' or np.any(times % 60 != 0) or
            (size > 1 and np.any(np.diff(times.astype(np.int64)) != 60))):
        raise ValueError('time must contain consecutive integer 1m candle opens; gaps refused')
    if times.dtype.kind == 'u' and size and times.max() > np.iinfo(np.int64).max - 60:
        raise ValueError('time outside supported integer UNIX seconds')
    times = times.astype(np.int64, copy=False)
    if size and (times.min() < 0 or times.max() > np.iinfo(np.int64).max - 60):
        raise ValueError('time outside supported integer UNIX seconds')
    arrays = {name: _numeric(df[name].to_numpy(), name, size) for name in required - {'time'}}
    if any(np.any(arrays[name] <= 0) for name in ('open', 'high', 'low', 'close')):
        raise ValueError('OHLC prices must be positive')
    if (np.any(arrays['volume'] < 0) or
            np.any(arrays['low'] > np.minimum(arrays['open'], arrays['close'])) or
            np.any(arrays['high'] < np.maximum(arrays['open'], arrays['close'])) or
            np.any(arrays['high'] < arrays['low'])):
        raise ValueError('invalid OHLC or volume')
    if not isinstance(signal, np.ndarray):
        raise ValueError('signal must be an ndarray')
    signal = _numeric(signal, 'signal', size)
    if not np.isin(signal, [-1, 0, 1]).all():
        raise ValueError('signal values must be -1, 0, or 1')
    for name in ('atr', 'ema9', 'momentum'):
        if name in df:
            arrays[name] = _numeric(df[name].to_numpy(), name, size, nan=True)
        else:
            arrays[name] = np.full(size, np.nan)
    if np.any(arrays['atr'] < 0) or np.any(arrays['ema9'] <= 0):
        raise ValueError('ATR must be nonnegative and EMA9 positive when present')
    momentum = arrays['momentum']
    if not np.isin(momentum[~np.isnan(momentum)], [-1, 0, 1]).all():
        raise ValueError('momentum must be -1, 0, or 1')
    if config.exit_mode == 'momentum' and not {'momentum', 'ema9'}.issubset(df.columns):
        raise ValueError('momentum mode requires momentum and ema9 columns')
    if config.deadfish_enabled:
        required_deadfish = {'deadfish_long', 'deadfish_short', 'strong_trend_long', 'strong_trend_short', 'momentum'}
        if not required_deadfish.issubset(df.columns):
            raise ValueError('deadfish requires joint weakening, strong trend and momentum feature flags')
        for name in required_deadfish - {'momentum'}:
            values = df[name].to_numpy()
            if values.dtype.kind not in 'biuf' or not np.isin(values, [0, 1]).all():
                raise ValueError(f'{name} must contain only 0 or 1')
            arrays[name] = values.astype(bool)
    if (config.exit_mode in {'partial', 'trailing'} or
            (config.exit_mode != 'original' and (stops is None or targets is None))) and 'atr' not in df:
        raise ValueError('ATR column required for ATR-based barriers')
    for name in ('exit_long', 'exit_short'):
        if name not in df:
            arrays[name] = np.zeros(size, dtype=bool)
        else:
            values = df[name].to_numpy()
            if values.shape != (size,) or values.dtype.kind not in 'biuf' or not np.isin(values, [0, 1]).all():
                raise ValueError(f'{name} must contain only 0 or 1')
            arrays[name] = values.astype(bool)
    proposals = []
    for name, value in (('stops', stops), ('targets', targets)):
        if value is None:
            proposals.append(None)
        else:
            if not isinstance(value, np.ndarray):
                raise ValueError(f'{name} must be an ndarray')
            values = _numeric(value, name, size, nan=True)
            # Zero/NaN represents no scanner proposal on a non-entry row.
            if np.any(values < 0):
                raise ValueError(f'{name} must be positive or missing')
            proposals.append(values)
    events = []
    if funding is not None:
        try:
            for event in funding:
                timestamp = _integer(event['time'], 'funding time')
                rate, mark = event['rate'], event['mark_price']
                if (isinstance(rate, (bool, np.bool_)) or isinstance(mark, (bool, np.bool_)) or
                        not isinstance(rate, Real) or not isinstance(mark, Real) or
                        not isfinite(rate) or not isfinite(mark) or mark <= 0):
                    raise ValueError('funding rate must be finite and mark_price positive')
                if events and timestamp <= events[-1][0]:
                    raise ValueError('funding settlements must be strictly increasing and unique')
                events.append((timestamp, float(rate), float(mark)))
        except (KeyError, TypeError) as exc:
            raise ValueError('funding must be a sequence of time/rate/mark_price mappings') from exc
    return start, end, times, arrays, signal, *proposals, events


def run(df, signal: np.ndarray, config: CostConfig, *, start: int, end: int,
        funding=None, stops=None, targets=None, record_equity_phases: bool = False) -> dict:
    """Execute external close-confirmed signals, with no IO or mutable globals.

    ``df.attrs['symbol']`` supplies the optional caller-owned symbol; a constant
    ``symbol`` column is also accepted. The external max-hold overlay applies to
    every mode, including original; it is not native strategy behavior.
    ``initial_risk_usdt`` is the sizing denominator times initial quantity,
    including conservative cost allowance, so net_r is a net return on risk.
    """
    if type(record_equity_phases) is not bool:
        raise ValueError('record_equity_phases must be boolean')
    start, end, times, a, signal, stops, targets, events = _inputs(
        df, signal, config, start, end, stops, targets, funding)
    symbol = df.attrs.get('symbol')
    if 'symbol' in df and len(df):
        if df['symbol'].isna().any() or df['symbol'].nunique() != 1:
            raise ValueError('one constant symbol per run is required')
        column_symbol = df['symbol'].iloc[0]
        if symbol is not None and symbol != column_symbol:
            raise ValueError('conflicting symbol attribute and column')
        symbol = column_symbol
    halfspread = config.spread_bps / 20000
    impact = halfspread + config.slippage
    cash = float(config.initial_equity)
    equity = [[start, cash]]
    equity_phases = ['initial']
    event_phase = 'initial'
    trades = []
    rejections = Counter()
    p = None
    last_exit_index = -config.cooldown_bars - 1
    event_index = bisect_left([e[0] for e in events], start)
    first = int(np.searchsorted(times, start, side='left'))
    final = int(np.searchsorted(times, end - 60, side='right')) - 1

    def mark(mid):
        if p is None:
            return cash
        actual = mid * (1 - p['sign'] * impact)
        return cash + p['remaining'] * (p['sign'] * (actual - p['entry']) - actual * config.fee)

    def profit_at(mid):
        actual = mid * (1 - p['sign'] * impact)
        return (p['gross_pnl'] - p['fees'] - p['funding_cost'] +
                p['remaining'] * (p['sign'] * (actual - p['entry']) - actual * config.fee))

    def observe(timestamp, mid=None):
        value = cash if p is None else mark(mid)
        if not isfinite(value):
            raise ValueError('non-finite equity from numeric overflow')
        equity.append([int(timestamp), float(value)])
        if record_equity_phases:
            equity_phases.append(event_phase)

    def settle(until, *, inclusive=False):
        nonlocal event_index, cash
        while event_index < len(events):
            timestamp, rate, mid = events[event_index]
            if timestamp > until or (timestamp == until and not inclusive):
                break
            if p is not None and timestamp >= p['entry_time']:
                debit = p['sign'] * p['remaining'] * mid * rate
                p['funding_cost'] += debit
                p['remaining_funding_cost'] += debit
                cash -= debit
                observe(timestamp, mid)
            event_index += 1

    def close(mid, quantity, timestamp, reason, index):
        nonlocal p, cash, last_exit_index
        actual = mid * (1 - p['sign'] * impact)
        gross = p['sign'] * quantity * (actual - p['entry'])
        fee = quantity * actual * config.fee
        cash += gross - fee
        p['gross_pnl'] += gross
        p['fees'] += fee
        p['spread_cost'] += quantity * mid * halfspread
        p['slippage_cost'] += quantity * mid * config.slippage
        p['exit_value'] += quantity * actual
        remaining_before = p['remaining']
        p['remaining'] -= quantity
        p['remaining_funding_cost'] *= p['remaining'] / remaining_before
        if p['remaining'] <= p['qty'] * 1e-12:
            trade = {name: p[name] for name in (
                'symbol', 'side', 'signal_time', 'entry_time', 'entry', 'qty',
                'notional', 'margin', 'initial_stop', 'initial_risk_usdt',
                'gross_pnl', 'fees', 'spread_cost', 'slippage_cost', 'funding_cost')}
            net = p['gross_pnl'] - p['fees'] - p['funding_cost']
            trade.update(exit_time=int(timestamp), exit=p['exit_value'] / p['qty'],
                         net_pnl=net, net_r=net / p['initial_risk_usdt'],
                         holding_seconds=int(timestamp - p['entry_time']), exit_reason=reason)
            if record_equity_phases:
                trade['exit_event_phase'] = event_phase
            trades.append(trade)
            p = None
            last_exit_index = index
            observe(timestamp)

    def roi_price(minutes):
        roi = .085 if minutes < 6 else .013 if minutes < 16 else .007 if minutes < 40 else 0.
        # Native leveraged *net* ROI: entry and exit fees are both included.
        target_per_unit = roi * p['entry'] / config.leverage + p['funding_cost'] / p['qty']
        actual = (p['entry'] * (p['sign'] + config.fee) + target_per_unit) / (p['sign'] - config.fee)
        return actual / (1 - p['sign'] * impact)

    for i in range(first, final + 1):
        event_phase = 'open'
        t = int(times[i])
        bar_close = t + 60
        opening, high, low, closing = (float(a[name][i]) for name in ('open', 'high', 'low', 'close'))
        occupied_at_open = p is not None
        if p is not None:
            s = p['sign']
            if s * (opening - p['liquidation']) <= 0:
                close(opening, p['remaining'], t, 'liquidation', i)
            elif s * (opening - p['stop']) <= 0:
                close(opening, p['remaining'], t, 'stop', i)
            else:
                exit_flag = config.use_source_exits and a['exit_long' if s == 1 else 'exit_short'][i - 1]
                if config.exit_mode == 'original':
                    exit_flag = exit_flag and profit_at(float(a['close'][i - 1])) > 0
                opposite_momentum = config.exit_mode == 'momentum' and (
                    a['momentum'][i - 1] == -s or s * (a['close'][i - 1] - a['ema9'][i - 1]) < 0)
                side_name = 'long' if s == 1 else 'short'
                deadfish = (config.deadfish_enabled and i >= 3 and
                            t - p['entry_time'] >= config.deadfish_min_hold_bars * 60 and
                            a['deadfish_' + side_name][i - 1] and
                            not a['strong_trend_' + side_name][i - 1] and
                            # Joint feature flag already requires three bars of
                            # fading momentum; EMA ordering need not reverse.
                            profit_at(float(a['close'][i - 1])) <
                            config.deadfish_stall_risk_fraction * p['initial_risk_usdt'])
                reason = ('exit_signal' if exit_flag else 'momentum' if opposite_momentum else
                          'deadfish' if deadfish else
                          'timeout' if t - p['entry_time'] >= config.max_hold_bars * 60 else None)
                if reason:
                    close(opening, p['remaining'], t, reason, i)
        if (p is None and not occupied_at_open and i > 0 and
                i > last_exit_index + config.cooldown_bars and times[i - 1] >= start and signal[i - 1]):
            s = int(signal[i - 1])
            entry = opening * (1 + s * impact)
            atr = float(a['atr'][i - 1])
            needs_atr = config.exit_mode in {'partial', 'trailing'} or (
                config.exit_mode != 'original' and (stops is None or targets is None))
            if needs_atr and (not isfinite(atr) or atr <= 0):
                rejections['invalid_atr'] += 1
            else:
                stop = (entry * (1 - s * .08 / config.leverage) if config.exit_mode == 'original' else
                        float(stops[i - 1]) if stops is not None else entry - s * atr * config.stop_atr)
                target = (np.nan if config.exit_mode == 'original' else
                          float(targets[i - 1]) if targets is not None else
                          entry + s * atr * (2 if config.exit_mode == 'partial' else config.target_atr))
                if (not isfinite(stop) or stop <= 0 or s * (entry - stop) <= 0 or
                        (config.exit_mode != 'original' and
                         (not isfinite(target) or target <= 0 or s * (target - entry) <= 0))):
                    rejections['invalid_barriers'] += 1
                elif (config.min_net_rr is not None and
                      net_reward_risk(opening, s, stop, target, config) < config.min_net_rr):
                    rejections['min_net_rr'] += 1
                else:
                    distance = abs(entry - stop)
                    # Retain the original conservative allowance, but bound it
                    # below by loss at the actual adverse stop fill and fees.
                    # A short stop's exit fee can exceed its entry fee.
                    stop_fill = stop * (1 - s * impact)
                    stop_loss = s * (entry - stop_fill) + config.fee * (entry + stop_fill)
                    risk_per_unit = max(distance + 2 * entry * (config.fee + impact), stop_loss)
                    qty = config.risk_usdt / risk_per_unit
                    entry_fee = qty * entry * config.fee
                    margin = qty * entry / config.leverage
                    if not all(isfinite(v) for v in (qty, margin, entry_fee)):
                        raise ValueError('non-finite sizing from numeric overflow')
                    if margin + entry_fee > cash:
                        rejections['insufficient_margin'] += 1
                    else:
                        liquidation = entry * (1 - s / config.leverage) / (1 - s * config.maintenance_fraction)
                        p = dict(symbol=symbol, side='LONG' if s == 1 else 'SHORT', sign=s,
                                 signal_time=int(times[i - 1]) + 60, entry_time=t, entry=entry,
                                 qty=qty, remaining=qty, notional=qty * entry, margin=margin,
                                 initial_stop=stop, stop=stop, target=target, atr=atr,
                                 initial_risk_usdt=qty * risk_per_unit, gross_pnl=0.,
                                 fees=entry_fee, spread_cost=qty * opening * halfspread,
                                 slippage_cost=qty * opening * config.slippage, funding_cost=0.,
                                 remaining_funding_cost=0., exit_value=0.,
                                 liquidation=liquidation, partial_done=False)
                        cash -= entry_fee
                        # Entry impact can put a valid proposal's stop beyond
                        # the current mid. The obtainable fill is the open,
                        # before any later intrabar barrier or profit exit.
                        if s * (opening - stop) <= 0:
                            close(opening, p['remaining'], t, 'stop', i)
        # ROI can fill at open after a schedule step, but before later OHLC.
        roi = None
        if p is not None and config.exit_mode == 'original' and signal[i - 1] != p['sign']:
            roi = roi_price((t - p['entry_time']) // 60)
            if p['sign'] * (opening - roi) >= 0:
                close(opening, p['remaining'], t, 'roi', i)
        if p is not None and config.exit_mode != 'original':
            s = p['sign']
            if config.exit_mode == 'partial' and not p['partial_done']:
                if s * (opening - p['entry'] - s * p['atr']) >= 0:
                    close(opening, p['qty'] / 2, t, 'partial', i)
                    p['partial_done'] = True
            if p is not None and s * (opening - p['target']) >= 0:
                close(opening, p['remaining'], t, 'target', i)
        settle(t, inclusive=True)
        # Timestamp alone cannot distinguish preceding candle close from the
        # following open/funding event at the same UTC month boundary.
        event_phase = 'bar_close'
        settle(bar_close)
        if p is not None and roi is not None:
            roi = roi_price((t - p['entry_time']) // 60)
        if p is not None:
            s = p['sign']
            adverse, favorable = (low, high) if s == 1 else (high, low)
            effective_stop = max(p['stop'], p['liquidation']) if s == 1 else min(p['stop'], p['liquidation'])
            stop_reason = 'stop' if effective_stop == p['stop'] else 'liquidation'
            adverse_mark = max(adverse, effective_stop) if s == 1 else min(adverse, effective_stop)
            if s * (adverse - effective_stop) <= 0:
                # SL-first is deliberately conservative when OHLC spans both
                # barriers. Do not invent peaks beyond a resting profit order.
                profit_limit = roi if config.exit_mode == 'original' else p['target']
                if config.exit_mode == 'partial' and not p['partial_done']:
                    profit_limit = p['entry'] + s * p['atr']
                favorable_mark = favorable
                if profit_limit is not None:
                    favorable_mark = min(favorable, profit_limit) if s == 1 else max(favorable, profit_limit)
                observe(bar_close, favorable_mark)
                observe(bar_close, adverse_mark)
                close(effective_stop, p['remaining'], bar_close, stop_reason, i)
            elif config.exit_mode == 'original':
                if roi is not None and s * (favorable - roi) >= 0:
                    observe(bar_close, adverse_mark)
                    close(roi, p['remaining'], bar_close, 'roi', i)
                else:
                    observe(bar_close, favorable)
                    observe(bar_close, adverse_mark)
            else:
                first_hit = False
                if config.exit_mode == 'partial' and not p['partial_done']:
                    first_target = p['entry'] + s * p['atr']
                    if s * (favorable - first_target) >= 0:
                        observe(bar_close, adverse_mark)
                        fill = max(opening, first_target) if s == 1 else min(opening, first_target)
                        close(fill, p['qty'] / 2, bar_close, 'partial', i)
                        p['partial_done'] = True
                        first_hit = True
                if p is not None and s * (favorable - p['target']) >= 0:
                    if not first_hit:
                        observe(bar_close, adverse_mark)
                    fill = max(opening, p['target']) if s == 1 else min(opening, p['target'])
                    close(fill, p['remaining'], bar_close, 'target', i)
                elif p is not None:
                    observe(bar_close, favorable)
                    observe(bar_close, adverse_mark)
        if p is not None:
            s = p['sign']
            if config.exit_mode == 'partial' and p['partial_done']:
                # Break-even for the REMAINING leg, including its entry and
                # exit fees, adverse spread/slippage and accrued funding.
                actual = (p['entry'] * (s + config.fee) +
                          p['remaining_funding_cost'] / p['remaining']) / (s - config.fee)
                next_stop = actual / (1 - s * impact)
                p['stop'] = max(p['stop'], next_stop) if s == 1 else min(p['stop'], next_stop)
            elif config.exit_mode == 'trailing':
                atr_now = float(a['atr'][i])
                if isfinite(atr_now) and atr_now > 0:
                    multiple = config.stop_atr if config.trailing_atr is None else config.trailing_atr
                    next_stop = (high - atr_now * multiple if s == 1 else low + atr_now * multiple)
                    p['stop'] = max(p['stop'], next_stop) if s == 1 else min(p['stop'], next_stop)
            elif config.exit_mode == 'original':
                favorable = high if s == 1 else low
                if profit_at(favorable) / p['margin'] >= .01:
                    next_stop = favorable * (1 - s * .005 / config.leverage)
                    p['stop'] = max(p['stop'], next_stop) if s == 1 else min(p['stop'], next_stop)
            if i == final:
                close(closing, p['remaining'], bar_close, 'terminal', i)
            else:
                observe(bar_close, closing)
        elif equity[-1][0] != bar_close:
            observe(bar_close)
    if trades and not np.isclose(cash, config.initial_equity + sum(t['net_pnl'] for t in trades),
                                 rtol=1e-10, atol=1e-8):
        raise ArithmeticError('final equity does not reconcile with trade net PnL')
    result = dict(trades=trades, equity=equity, rejections=dict(rejections))
    if record_equity_phases:
        result['equity_phases'] = equity_phases
    return result
