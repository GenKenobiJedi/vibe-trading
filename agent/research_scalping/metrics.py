"""Deterministic arithmetic summaries of reconciled research trade/equity sets.

Callers must supply equity for the SAME subset of trades. A full portfolio curve
cannot describe drawdown for a filtered coin, month, side, strategy, or window.
The final equity reconciliation catches many accidental mismatches; it cannot
establish that intermediate marks describe the correct subset.
"""
from __future__ import annotations

from collections import Counter
from math import fsum, isclose, isfinite
from numbers import Real
from statistics import median


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def summarize(trades, equity, initial_equity=1000., *, allow_nonpositive_initial=False) -> dict:
    """Return net performance, cost totals and equity-based drawdown percentages.

    Returns ``N`` and ``trades`` as identical count aliases. Undefined profit
    factor is JSON-null, with ``profit_factor_status`` equal to ``no_trades`` or
    ``no_losses``; finite profit factor has status ``ok``. Empty samples have
    null win rate, zero PnL and zero drawdown. No annualization is inferred.
    """
    initial_equity = _number(initial_equity, 'initial_equity')
    if initial_equity <= 0 and not allow_nonpositive_initial:
        raise ValueError('initial_equity must be positive')
    trades = list(trades)
    values = {name: [] for name in ('net_pnl', 'holding_seconds', 'net_r', 'fees',
                                    'slippage_cost', 'spread_cost', 'funding_cost')}
    reasons = Counter()
    for trade in trades:
        try:
            for name in values:
                values[name].append(_number(trade[name], name))
            reason = trade['exit_reason']
            if not isinstance(reason, str) or not reason:
                raise ValueError('exit_reason must be a nonempty string')
            reasons[reason] += 1
        except (KeyError, TypeError) as exc:
            raise ValueError('malformed trade record') from exc
    if any(v < 0 for name in ('holding_seconds', 'fees', 'spread_cost', 'slippage_cost') for v in values[name]):
        raise ValueError('holding duration and execution costs must be nonnegative')
    points = []
    previous_time = None
    for point in equity:
        try:
            if len(point) != 2:
                raise ValueError('equity points must contain timestamp and value')
            timestamp, amount = _number(point[0], 'equity timestamp'), _number(point[1], 'equity')
        except TypeError as exc:
            raise ValueError('malformed equity point') from exc
        if previous_time is not None and timestamp < previous_time:
            raise ValueError('equity timestamps must be nondecreasing')
        points.append(amount)
        previous_time = timestamp
    net = fsum(values['net_pnl'])
    if trades and not points:
        raise ValueError('equity required to reconcile nonempty trades')
    if points and not isclose(points[-1], initial_equity + net, rel_tol=1e-10, abs_tol=1e-8):
        raise ValueError('equity does not reconcile with initial equity plus trade net PnL')
    count = len(trades)
    gains = fsum(v for v in values['net_pnl'] if v > 0)
    losses = -fsum(v for v in values['net_pnl'] if v < 0)
    peak, drawdown, drawdown_usdt = initial_equity, 0., 0.
    if count:
        for amount in points:
            peak = max(peak, amount)
            drawdown_usdt = max(drawdown_usdt, peak - amount)
            if initial_equity > 0:
                drawdown = max(drawdown, (peak - amount) / peak * 100)
    return dict(N=count, trades=count,
                win_rate=sum(v > 0 for v in values['net_pnl']) / count if count else None,
                net_pnl=net, return_pct=net / initial_equity * 100 if initial_equity > 0 else None,
                equity_percentage_status='ok' if initial_equity > 0 else 'nonpositive_initial_equity',
                profit_factor=gains / losses if losses else None,
                profit_factor_status='ok' if losses else 'no_losses' if count else 'no_trades',
                max_drawdown_pct=drawdown if initial_equity > 0 else None,
                max_drawdown_usdt=drawdown_usdt,
                avg_hold_seconds=fsum(values['holding_seconds']) / count if count else None,
                median_hold_seconds=median(values['holding_seconds']) if count else None,
                avg_net_r=fsum(values['net_r']) / count if count else None,
                fees=fsum(values['fees']), slippage_cost=fsum(values['slippage_cost']),
                spread_cost=fsum(values['spread_cost']), funding_cost=fsum(values['funding_cost']),
                exit_reasons=dict(sorted(reasons.items())))
