"""Offline, split-aware accounting and durable research artifacts.

Monthly marked-to-market change is separate from whole-trade exit attribution.
Direction curves contain realized PnL only and never claim direction MTM risk.
"""
from __future__ import annotations

from collections import Counter
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from research_scalping.metrics import summarize


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')


def canonical(payload: object) -> bytes:
    """Normalize tuples/lists identically and reject non-finite JSON numbers."""
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False,
                      default=_json_default).encode()


def digest(payload: object) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


def file_hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _durable(path: Path, raw: bytes, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    with pending.open('wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if exclusive:
            os.link(pending, path)
            pending.unlink()
        else:
            pending.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        pending.unlink(missing_ok=True)


def atomic_json(path: Path, payload: object) -> None:
    raw = canonical(payload)
    _durable(path, gzip.compress(raw, mtime=0) if path.suffix == '.gz' else raw)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    """Write structured values as JSON cells, not Python repr strings."""
    stream = io.StringIO(newline='')
    fields = sorted({key for row in rows for key in row})
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: canonical(value).decode() if isinstance(value, (list, dict)) else value
                         for key, value in row.items()})
    _durable(path, stream.getvalue().encode())


def load_json(path: Path) -> object:
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def seal(path: Path, payload: dict) -> str:
    """Atomically create an immutable, canonical parent-linkable decision."""
    checksum = digest(payload)
    _durable(path, canonical({'payload': payload, 'sha256': checksum}), exclusive=True)
    return checksum


def verify_seal(path: Path) -> dict:
    record = load_json(path)
    if not isinstance(record, dict) or digest(record.get('payload')) != record.get('sha256'):
        raise ValueError(f'Invalid or modified seal: {path.name}')
    return record['payload']


def months(start: int, end: int) -> Iterable[tuple[str, int, int]]:
    cursor = dt.datetime.fromtimestamp(start, dt.timezone.utc).replace(day=1, hour=0, minute=0, second=0)
    while cursor.timestamp() < end:
        following = cursor.replace(year=cursor.year + (cursor.month == 12), month=cursor.month % 12 + 1)
        yield cursor.strftime('%Y-%m'), int(cursor.timestamp()), int(following.timestamp())
        cursor = following


def drawdown(equity: list, initial: float) -> tuple[float | None, float]:
    peak, dollars, percent = initial, 0., 0.
    for _, value in equity:
        peak = max(peak, value)
        dollars = max(dollars, peak - value)
        if initial > 0:
            percent = max(percent, (peak - value) / peak * 100.)
    return (percent if initial > 0 else None), dollars


def realized(trades: list[dict], initial: float, start: int) -> list:
    cash, curve = initial, [[start, initial]]
    for trade in sorted(trades, key=lambda t: (t['exit_time'], t['entry_time'])):
        cash += trade['net_pnl']
        curve.append([trade['exit_time'], cash])
    return curve


def metrics(trades: list[dict], equity: list, initial: float = 1000.) -> dict:
    result = summarize(trades, equity, initial_equity=initial, allow_nonpositive_initial=initial <= 0)
    result['max_drawdown_pct'], result['max_drawdown_usdt'] = drawdown(equity, initial)
    return result


def _period_curve(result: dict, lo: int, hi: int, initial: float) -> tuple:
    """Separate prior candle-close observations from next-open events at midnight."""
    equity = result['equity']
    phases = result.get('equity_phases')
    if phases is None:
        for boundary in (lo, hi):
            if len({value for timestamp, value in equity if timestamp == boundary}) > 1:
                raise ValueError('Equity event phase evidence required for ambiguous monthly boundary')
        prior = [value for timestamp, value in equity if timestamp <= lo]
        base = prior[-1] if prior else initial
        return base, [[lo, base], *[point for point in equity if lo < point[0] <= hi]]
    if (not isinstance(phases, list) or len(phases) != len(equity) or
            any(phase not in ('initial', 'open', 'bar_close') for phase in phases)):
        raise ValueError('Valid equity event phase per observation required')
    prior = [value for (timestamp, value), phase in zip(equity, phases)
             if timestamp < lo or (timestamp == lo and phase != 'open')]
    base = prior[-1] if prior else initial
    curve = [[lo, base], *[point for point, phase in zip(equity, phases)
             if lo < point[0] < hi or (point[0] == lo and phase == 'open') or
             (point[0] == hi and phase == 'bar_close')]]
    return base, curve


def groups(result: dict, metadata: dict) -> list[dict]:
    """Produce every coin/month/direction cell without assigning full MTM to sides."""
    trades, equity = result['trades'], result['equity']
    start, end = metadata['window_start'], metadata['window_end']
    initial = metadata.get('initial_equity', 1000.)
    internal_boundaries = {lo for _, lo, _ in months(start, end) if start < lo < end}
    if any(trade['exit_time'] in internal_boundaries and
           trade.get('exit_event_phase') not in ('open', 'bar_close') for trade in trades):
        raise ValueError('Valid exit event phase required at an internal monthly boundary')
    allocated = lambda t: min(t['exit_time'] - int(t.get('exit_event_phase') == 'bar_close'), end - 1)
    rows = []
    for month, lo, hi in [('ALL', start, end), *months(start, end)]:
        lo, hi = max(lo, start), min(hi, end)
        subset = trades if month == 'ALL' else [t for t in trades if lo <= allocated(t) < hi]
        # A close observation and subsequent open/funding can share a timestamp.
        # Preserve their recorded phase; timestamps alone do not establish month.
        base, curve = (initial, equity) if month == 'ALL' else _period_curve(result, lo, hi, initial)
        summary = metrics(subset, realized(subset, base, lo), base)
        summary['closed_trade_return_pct'] = summary['return_pct']
        summary['mtm_net_pnl'] = curve[-1][1] - base
        summary['return_pct'] = summary['mtm_net_pnl'] / base * 100. if base > 0 else None
        summary['max_drawdown_pct'], summary['max_drawdown_usdt'] = drawdown(curve, base)
        rows.append({**metadata, 'status': 'EVALUATED', 'month': month, 'direction': 'BOTH',
                     'pnl_attribution': 'net_pnl by execution-phase exit month; mtm_net_pnl by phase-aware boundary marks',
                     'drawdown_basis': 'marked_to_market_including_adverse_extrema', **summary})
        for side in ('LONG', 'SHORT'):
            previous = [t for t in trades if t['side'] == side and allocated(t) < lo]
            side_base = initial + math.fsum(t['net_pnl'] for t in previous)
            side_trades = [t for t in subset if t['side'] == side]
            summary = metrics(side_trades, realized(side_trades, side_base, lo), side_base)
            summary['realized_drawdown_pct'] = summary.pop('max_drawdown_pct')
            summary['realized_drawdown_usdt'] = summary.pop('max_drawdown_usdt')
            rows.append({**metadata, 'status': 'EVALUATED', 'month': month, 'direction': side,
                         **summary, 'max_drawdown_pct': None, 'max_drawdown_usdt': None,
                         'mtm_net_pnl': None, 'pnl_attribution': 'whole-trade net PnL by execution-phase exit month',
                         'drawdown_basis': 'realized-only; direction MTM unavailable'})
    return rows


def unavailable(metadata: dict, reason: str) -> list[dict]:
    empty = metrics([], [], 1000.)
    empty = {key: None for key in empty}
    return [{**metadata, **empty, 'month': month, 'direction': direction,
             'status': 'NOT_EVALUABLE', 'reason': reason, 'mtm_net_pnl': None,
             'drawdown_basis': 'unavailable candles'}
            for month in ['ALL', *(m for m, _, _ in months(metadata['window_start'], metadata['window_end']))]
            for direction in ('BOTH', 'LONG', 'SHORT')]


def combine(paths: Iterable[Path]) -> dict:
    """Pool independent sleeves with one equity series and one ledger loaded at a time.

    Close-equity aggregation cannot synchronize per-coin adverse extrema. These
    remain in coin groups. Durations are a bounded histogram, not pooled trades.
    """
    aggregate = None
    initial = 0.
    count = wins = 0
    gains = losses = net = net_r = 0.
    durations, reasons, rejections = Counter(), Counter(), Counter()
    totals = dict.fromkeys(('fees', 'slippage_cost', 'spread_cost', 'funding_cost'), 0.)
    sleeves = 0
    for path in paths:
        result = load_json(path)
        opening = result.get('initial_equity', 1000.)
        series = pd.Series([p[1] for p in result['equity']], index=[p[0] for p in result['equity']])
        series = series.groupby(level=0).last()
        if aggregate is None:
            aggregate = series
        else:
            pair = pd.concat([aggregate, series], axis=1).sort_index().ffill()
            pair.iloc[:, 0] = pair.iloc[:, 0].fillna(initial)
            pair.iloc[:, 1] = pair.iloc[:, 1].fillna(opening)
            aggregate = pair.sum(axis=1)
        initial += opening
        sleeves += 1
        rejections.update(result.get('rejections', {}))
        for trade in result['trades']:
            pnl = trade['net_pnl']
            count += 1
            wins += pnl > 0
            gains += max(pnl, 0.)
            losses += max(-pnl, 0.)
            net += pnl
            net_r += trade['net_r']
            durations[trade['holding_seconds']] += 1
            reasons[trade['exit_reason']] += 1
            for key in totals:
                totals[key] += trade[key]
    if aggregate is None:
        return {**metrics([], []), 'coin_sleeves': 0, 'status': 'NOT_EVALUABLE'}
    if not math.isclose(aggregate.iloc[-1], initial + net, rel_tol=1e-10, abs_tol=1e-7):
        raise ArithmeticError('Pooled close equity does not reconcile')
    pct, dollars = drawdown(list(aggregate.items()), initial)
    rank_values = []
    cumulative = 0
    for duration, number in sorted(durations.items()):
        before, cumulative = cumulative, cumulative + number
        for rank in ((count - 1) // 2, count // 2):
            if before <= rank < cumulative:
                rank_values.append(duration)
    return dict(N=count, trades=count, win_rate=wins/count if count else None,
                net_pnl=net, profit_factor=gains/losses if losses else None,
                profit_factor_status='ok' if losses else 'no_losses' if count else 'no_trades',
                return_pct=net/initial*100 if initial > 0 else None,
                max_drawdown_pct=pct, max_drawdown_usdt=dollars,
                avg_net_r=net_r/count if count else None,
                avg_hold_seconds=sum(k*v for k,v in durations.items())/count if count else None,
                median_hold_seconds=sum(rank_values)/2 if count else None,
                exit_reasons=dict(reasons), rejections=dict(rejections), **totals,
                coin_sleeves=sleeves, status='EVALUATED',
                drawdown_basis='sum of independent sleeve close equity; coin adverse extrema separate')
