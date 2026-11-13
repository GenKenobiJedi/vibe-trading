"""Offline execution contract tests; all market inputs are synthetic."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from research_scalping.execution import CostConfig, run


def bars(n=8, price=100., atr=1.):
    return pd.DataFrame(dict(time=np.arange(n, dtype=np.int64) * 60,
                             open=price, high=price, low=price, close=price,
                             volume=10., atr=atr, momentum=0, ema9=price))


def config(**kwargs):
    options = dict(fee=0., slippage=0., spread_bps=0., cooldown_bars=0, max_hold_bars=100)
    options.update(kwargs)
    return replace(CostConfig(), **options)


def execute(frame, side=1, *, at=0, settings=None, **kwargs):
    signal = np.zeros(len(frame), dtype=np.int8)
    signal[at] = side
    return run(frame, signal, settings or config(), start=0,
               end=int(frame.time.iloc[-1]) + 60, **kwargs)


def test_frozen_defaults_and_exact_interface():
    c = CostConfig()
    assert (c.fee, c.slippage, c.spread_bps, c.leverage, c.risk_usdt,
            c.initial_equity, c.max_hold_bars, c.cooldown_bars,
            c.stop_atr, c.target_atr, c.exit_mode, c.maintenance_fraction) == (
                .0005, .0002, 2., 10., 10., 1000., 15, 2, 1.2, 1.8, 'atr', .005)
    with pytest.raises(FrozenInstanceError):
        c.fee = 0


def test_signal_close_next_open_not_signal_bar_and_terminal_costs():
    df = bars(4)
    df.loc[0, ['high', 'low']] = [200., 1.]
    df.loc[1:, ['open', 'high', 'low', 'close']] = 110.
    df.attrs['symbol'] = 'SYNTH/USDT'
    result = execute(df)
    trade, = result['trades']
    assert set(result) == {'trades', 'equity', 'rejections'}
    assert (trade['signal_time'], trade['entry_time'], trade['exit_time']) == (60, 60, 240)
    assert trade['entry'] == trade['exit'] == 110.
    assert trade['symbol'] == 'SYNTH/USDT'
    assert trade['exit_reason'] == 'terminal'
    assert result['equity'][0] == [0, 1000.]
    assert result['equity'][-1] == [240, 1000.]


def test_window_starts_flat_no_pre_window_signal_or_outside_fill():
    df = bars()
    signal = np.ones(len(df), dtype=np.int8)
    out = run(df, signal, config(), start=120, end=300)
    trade, = out['trades']
    assert trade['entry_time'] == 180
    assert trade['signal_time'] == 180
    assert trade['exit_time'] == 300
    signal[:] = 0
    signal[1] = signal[4] = 1
    assert run(df, signal, config(), start=120, end=300)['trades'] == []


@pytest.mark.parametrize('side', [1, -1])
def test_same_entry_bar_stop_first_when_stop_and_target_hit(side):
    df = bars(3)
    df.loc[1, ['high', 'low']] = [105., 95.]
    trade, = execute(df, side)['trades']
    assert trade['exit_reason'] == 'stop'
    assert trade['exit'] == pytest.approx(100 - side * 1.2)
    assert trade['gross_pnl'] == pytest.approx(-10)


@pytest.mark.parametrize('side', [1, -1])
def test_costs_actual_prices_single_fee_and_no_leverage_pnl_multiplier(side):
    df = bars(4, atr=10.)
    c = CostConfig(fee=.001, slippage=.002, spread_bps=20.,
                   stop_atr=1., target_atr=2., max_hold_bars=100)
    trade, = execute(df, side, settings=c)['trades']
    entry = 100 * (1 + side * .003)
    exit_price = 100 * (1 - side * .003)
    quantity = 10 / (10 + 2 * entry * .004)
    assert trade['entry'] == pytest.approx(entry)
    assert trade['exit'] == pytest.approx(exit_price)
    assert trade['qty'] == pytest.approx(quantity)
    assert trade['margin'] == pytest.approx(quantity * entry / 10)
    assert trade['notional'] == pytest.approx(quantity * entry)
    assert trade['initial_risk_usdt'] == pytest.approx(10)
    assert trade['fees'] == pytest.approx(quantity * (entry + exit_price) * .001)
    assert trade['spread_cost'] == pytest.approx(quantity * 200 * .001)
    assert trade['slippage_cost'] == pytest.approx(quantity * 200 * .002)
    assert trade['gross_pnl'] == pytest.approx(quantity * side * (exit_price - entry))
    assert trade['net_pnl'] == pytest.approx(trade['gross_pnl'] - trade['fees'])
    other, = execute(df, side, settings=replace(c, leverage=5.))['trades']
    assert other['net_pnl'] == pytest.approx(trade['net_pnl'])
    assert other['margin'] == pytest.approx(trade['margin'] * 2)


def test_margin_rejection_no_silent_downsize():
    result = execute(bars(3, atr=.01), settings=config(leverage=1., risk_usdt=100.))
    assert result['trades'] == []
    assert result['rejections']['insufficient_margin'] == 1


@pytest.mark.parametrize('side', [1, -1])
def test_gap_stop_precedes_scheduled_exit_and_uses_adverse_open(side):
    df = bars(4)
    df['exit_long'] = df['exit_short'] = 0
    df.loc[1, ['exit_long', 'exit_short']] = 1
    df.loc[2:, ['open', 'high', 'low', 'close']] = 100 - side * 5
    trade, = execute(df, side)['trades']
    assert (trade['exit_time'], trade['exit_reason']) == (120, 'stop')
    assert trade['exit'] == 100 - side * 5


def test_scheduled_exit_open_precedes_intrabar_stop():
    df = bars(4)
    df['exit_long'] = [0, 1, 0, 0]
    df.loc[2, ['high', 'low']] = [104., 90.]
    trade, = execute(df)['trades']
    assert (trade['exit_time'], trade['exit'], trade['exit_reason']) == (120, 100., 'exit_signal')


@pytest.mark.parametrize('mode', ['trailing', 'original'])
def test_trailing_closed_bar_ratchet_only_applies_next_bar(mode):
    df = bars(4)
    df.loc[1, ['high', 'low', 'close']] = [100.3, 99.9, 100.2]
    df.loc[2, ['open', 'high', 'low', 'close']] = [100.28, 100.28, 98.9, 100.]
    c = config(exit_mode=mode)
    trade, = execute(df, settings=c)['trades']
    assert trade['exit_time'] >= 180
    assert trade['exit_reason'] == 'stop'
    expected = 99.1 if mode == 'trailing' else 100.3 * (1 - .005 / 10)
    assert trade['exit'] == pytest.approx(expected)


@pytest.mark.parametrize('side', [1, -1])
def test_partial_uses_original_half_quantities_and_weighted_exit(side):
    df = bars(4)
    df.loc[1:, ['high' if side == 1 else 'low']] = 100 + side * 1.1
    df.loc[2:, ['open', 'high', 'low', 'close']] = 100 + side * 2.
    trade, = execute(df, side, settings=config(exit_mode='partial'))['trades']
    assert trade['exit_reason'] == 'target'
    assert trade['exit'] == pytest.approx(100 + side * 1.5)
    assert trade['gross_pnl'] == pytest.approx(trade['qty'] * 1.5)


def test_partial_break_even_delayed_and_cost_aware():
    df = bars(4)
    df.loc[1, ['high', 'low', 'close']] = [101.1, 99.8, 100.5]
    df.loc[2, ['open', 'high', 'low', 'close']] = [100.5, 100.5, 99.8, 100.]
    c = replace(config(exit_mode='partial'), fee=.001, slippage=.0002, spread_bps=2.)
    trade, = execute(df, settings=c)['trades']
    assert trade['exit_reason'] == 'stop'
    assert trade['exit_time'] == 180
    half = trade['qty'] / 2
    first_exit = (trade['entry'] + 1) * (1 - .0003)
    # The remaining half's break-even includes its allocated entry fee.
    remaining_exit = trade['entry'] * (1 + c.fee) / (1 - c.fee)
    assert trade['exit'] == pytest.approx((first_exit + remaining_exit) / 2)
    assert trade['fees'] == pytest.approx(trade['qty'] * trade['entry'] * c.fee +
                                         half * (first_exit + remaining_exit) * c.fee)


@pytest.mark.parametrize('side', [1, -1])
def test_funding_real_settlements_signed_and_remaining_quantity(side):
    df = bars(5)
    df.loc[1, 'high' if side == 1 else 'low'] = 100 + side * 1.1
    df.loc[2:, ['open', 'high', 'low', 'close']] = 100 + side * .5
    funding = [dict(time=30, rate=.5, mark_price=100.),
               dict(time=90, rate=.001, mark_price=101.),
               dict(time=150, rate=-.002, mark_price=99.),
               dict(time=300, rate=1., mark_price=100.)]
    result = execute(df, side, settings=config(exit_mode='partial'), funding=funding)
    trade, = result['trades']
    expected = side * trade['qty'] * (101 * .001 + .5 * 99 * -.002)
    assert trade['funding_cost'] == pytest.approx(expected)
    assert trade['net_pnl'] == pytest.approx(trade['gross_pnl'] - expected)
    assert result['equity'][-1][1] == pytest.approx(1000 + trade['net_pnl'])


@pytest.mark.parametrize('trigger', ['momentum', 'ema9'])
@pytest.mark.parametrize('side', [1, -1])
def test_momentum_exit_only_next_open(trigger, side):
    df = bars(5)
    df.loc[2, trigger] = -side if trigger == 'momentum' else 100 + side
    df.loc[3:, ['open', 'high', 'low', 'close']] = 100 + side * .5
    trade, = execute(df, side, settings=config(exit_mode='momentum'))['trades']
    assert (trade['exit_time'], trade['exit_reason']) == (180, 'momentum')
    assert trade['exit'] == 100 + side * .5


def test_timeout_is_external_open_overlay_and_cooldown_blocks_full_bars():
    df = bars(10)
    out = run(df, np.ones(len(df), dtype=np.int8),
              config(max_hold_bars=2, cooldown_bars=2), start=0, end=600)
    assert [t['entry_time'] for t in out['trades']] == [60, 360]
    assert [t['exit_time'] for t in out['trades']] == [180, 480]
    assert all(t['exit_reason'] == 'timeout' for t in out['trades'])


def test_no_same_bar_reentry_from_stale_signal_after_intrabar_stop():
    df = bars(4)
    df.loc[1, ['high', 'low']] = [103., 97.]
    out = run(df, np.ones(4, dtype=np.int8), config(), start=0, end=240)
    assert [t['entry_time'] for t in out['trades']] == [60, 120]


@pytest.mark.parametrize('leverage', [5., 10., 20.])
def test_original_stop_is_leveraged_roi_not_eight_percent_price(leverage):
    df = bars(3)
    df.loc[1, 'low'] = 97.
    trade, = execute(df, settings=config(exit_mode='original', leverage=leverage))['trades']
    assert trade['initial_stop'] == pytest.approx(100 * (1 - .08 / leverage))
    assert trade['exit'] == pytest.approx(trade['initial_stop'])


def test_original_roi_schedule_and_ignore_prior_entry():
    df = bars(10)
    df.loc[7:, ['open', 'high', 'low', 'close']] = 100.14
    trade, = execute(df, settings=config(exit_mode='original'))['trades']
    assert trade['exit_reason'] == 'roi'
    assert trade['exit_time'] == 420  # A schedule change is actionable at this open.
    assert trade['exit'] == pytest.approx(100.14)
    sig = np.zeros(10, dtype=np.int8)
    sig[0] = sig[6] = sig[7] = sig[8] = 1
    other, = run(df, sig, config(exit_mode='original'), start=0, end=600)['trades']
    assert other['exit_reason'] == 'terminal'


def test_original_exit_signal_profit_only_then_next_open():
    df = bars(5)
    df['exit_long'] = [0, 1, 1, 0, 0]
    df.loc[1, ['open', 'high', 'low', 'close']] = [100., 100., 99.8, 99.9]
    df.loc[2:, ['open', 'high', 'low', 'close']] = 100.02
    trade, = execute(df, settings=config(exit_mode='original'))['trades']
    assert (trade['exit_time'], trade['exit_reason']) == (180, 'exit_signal')


def test_original_roi_threshold_accounts_for_both_fees():
    df = bars(2)
    df.loc[1, 'high'] = 100.86  # Gross 8.6% margin ROI; net below 8.5%.
    trade, = execute(df, settings=replace(config(exit_mode='original'), fee=.001))['trades']
    assert trade['exit_reason'] == 'terminal'


def test_liquidation_guard_is_assumed_not_exchange_model():
    df = bars(3, atr=20.)
    df.loc[1, 'low'] = 80.
    trade, = execute(df, settings=config())['trades']
    assert trade['exit_reason'] == 'liquidation'
    assert trade['exit'] == pytest.approx(100 * .9 / .995)


def test_mtm_extrema_do_not_realize_hypothetical_loss_and_final_reconciles():
    df = bars(4, atr=10.)
    df.loc[1, ['high', 'low', 'close']] = [102., 95., 101.]
    df.loc[2:, ['open', 'high', 'low', 'close']] = 102.
    result = execute(df)
    trade, = result['trades']
    assert min(v for _, v in result['equity']) == pytest.approx(1000 - 5 * trade['qty'])
    assert max(v for _, v in result['equity']) == pytest.approx(1000 + 2 * trade['qty'])
    assert result['equity'][-1][1] == pytest.approx(1000 + trade['net_pnl'])
    assert all(a[0] <= b[0] for a, b in zip(result['equity'], result['equity'][1:]))


@pytest.mark.parametrize('mode', ['atr', 'partial', 'trailing', 'momentum', 'original'])
def test_deterministic_many_trades_equity_reconciliation_and_no_mutation(mode):
    rng = np.random.default_rng(37)
    df = bars(180)
    values = 100 + rng.normal(0, .4, len(df))
    df['open'] = values
    df['close'] = values + rng.normal(0, .1, len(df))
    df['high'] = df[['open', 'close']].max(axis=1) + .3
    df['low'] = df[['open', 'close']].min(axis=1) - .3
    df['momentum'] = rng.choice([-1, 0, 1], len(df))
    signal = rng.choice([-1, 0, 1], len(df)).astype(np.int8)
    before, signal_before = df.copy(deep=True), signal.copy()
    c = CostConfig(exit_mode=mode, max_hold_bars=7, cooldown_bars=1)
    funding = [dict(time=601, rate=.001, mark_price=100.),
               dict(time=3001, rate=-.0003, mark_price=100.)]
    result = run(df, signal, c, start=0, end=10800, funding=funding)
    assert result == run(df, signal, c, start=0, end=10800, funding=funding)
    assert result['equity'][-1][1] == pytest.approx(
        c.initial_equity + sum(t['net_pnl'] for t in result['trades']))
    for trade in result['trades']:
        assert trade['net_pnl'] == pytest.approx(trade['gross_pnl'] - trade['fees'] - trade['funding_cost'])
        assert trade['net_r'] == pytest.approx(trade['net_pnl'] / trade['initial_risk_usdt'])
    pd.testing.assert_frame_equal(df, before)
    np.testing.assert_array_equal(signal, signal_before)


def test_scanner_barriers_read_at_signal_index_not_fill_index():
    df = bars(4)
    stops = np.array([99., 50., 50., 50.])
    targets = np.array([102., 150., 150., 150.])
    df.loc[1, 'high'] = 103.
    trade, = execute(df, stops=stops, targets=targets)['trades']
    assert (trade['initial_stop'], trade['exit']) == (99., 102.)


@pytest.mark.parametrize('change', [
    lambda d: d.drop(columns='open'),
    lambda d: d.set_axis([1, 2, 3]),
    lambda d: d.assign(time=[0, 60, 180]),
    lambda d: d.assign(time=[0., 60., 120.]),
    lambda d: d.assign(high=99.),
    lambda d: d.assign(low=np.nan),
    lambda d: d.assign(volume=-1.),
])
def test_malformed_frame_refused(change):
    with pytest.raises(ValueError):
        execute(change(bars(3)))


@pytest.mark.parametrize('signal', [np.array([1, 2, 0]), np.array([1., np.nan, 0.]),
                                   np.array([[1, 0, 0]]), np.array([1, 0])])
def test_malformed_signals_refused(signal):
    with pytest.raises(ValueError):
        run(bars(3), signal, config(), start=0, end=180)


@pytest.mark.parametrize('kwargs', [dict(fee=-1), dict(slippage=float('nan')),
                                   dict(leverage=0), dict(risk_usdt=0),
                                   dict(max_hold_bars=0), dict(cooldown_bars=-1),
                                   dict(exit_mode='unknown'), dict(maintenance_fraction=1.)])
def test_invalid_configuration_refused(kwargs):
    with pytest.raises(ValueError):
        execute(bars(3), settings=replace(CostConfig(), **kwargs))


@pytest.mark.parametrize('kwargs', [dict(start=120, end=60), dict(start=0., end=180),
                                   dict(start=0, end=180, stops=np.array([99.])),
                                   dict(start=0, end=180, funding=[dict(time=90, rate=np.nan, mark_price=100)]),
                                   dict(start=0, end=180, funding=[dict(time=90, rate=.001, mark_price=0)])])
def test_invalid_window_barriers_or_funding_refused(kwargs):
    with pytest.raises(ValueError):
        run(bars(3), np.array([1, 0, 0]), config(), **kwargs)


def test_nan_warmup_atr_skips_entry_with_rejection():
    result = execute(bars(3, atr=np.nan))
    assert result['trades'] == []
    assert result['rejections']['invalid_atr'] == 1


@pytest.mark.parametrize('rate,price,reason', [(.0005, 100.03, 'terminal'),
                                             (-.0005, 99.97, 'exit_signal')])
def test_original_profit_only_includes_signed_funding(rate, price, reason):
    df = bars(4)
    df['exit_long'] = [0, 1, 0, 0]
    df.loc[1, ['high', 'low', 'close']] = [max(100., price), min(100., price), price]
    trade, = execute(df, settings=config(exit_mode='original'),
                     funding=[dict(time=90, rate=rate, mark_price=100.)])['trades']
    assert trade['exit_reason'] == reason


def test_original_trailing_activation_requires_net_leveraged_profit():
    df = bars(3)
    df.loc[1, 'high'] = 100.15
    df.loc[2, 'low'] = 99.5
    c = replace(config(exit_mode='original'), fee=.001)
    trade, = execute(df, settings=c)['trades']
    assert trade['exit_reason'] == 'terminal'


@pytest.mark.parametrize('side', [1, -1])
@pytest.mark.parametrize('mode', ['atr', 'partial'])
def test_carried_gap_target_executes_at_open_before_later_stop(side, mode):
    df = bars(4)
    df.loc[2, ['open', 'high', 'low', 'close']] = (
        [103., 103., 90., 100.] if side == 1 else [97., 110., 97., 100.])
    trade, = execute(df, side, settings=config(exit_mode=mode))['trades']
    assert (trade['exit_time'], trade['exit_reason']) == (120, 'target')
    assert trade['exit'] == 100 + side * 3


def test_open_roi_exit_excludes_same_timestamp_funding():
    df = bars(9)
    df.loc[7:, ['open', 'high', 'low', 'close']] = 100.14
    trade, = execute(df, settings=config(exit_mode='original'),
                     funding=[dict(time=420, rate=-.00001, mark_price=100.)])['trades']
    assert trade['exit_reason'] == 'roi'
    assert trade['exit_time'] == 420
    assert trade['funding_cost'] == 0


def test_funding_at_entry_included_but_after_terminal_excluded():
    result = execute(bars(3), funding=[dict(time=60, rate=.001, mark_price=100.),
                                      dict(time=180, rate=.5, mark_price=100.)])
    trade, = result['trades']
    assert trade['funding_cost'] == pytest.approx(trade['qty'] * .1)


def test_empty_window_and_empty_input_have_initial_equity_only():
    df = bars(3)
    result = run(df, np.zeros(3, dtype=np.int8), config(), start=600, end=900)
    assert result == dict(trades=[], equity=[[600, 1000.]], rejections={})
    df = bars(0)
    assert run(df, np.zeros(0, dtype=np.int8), config(), start=0, end=60)['trades'] == []


def test_signal_future_suffix_cannot_change_already_closed_trade():
    df = bars(8)
    df.loc[1, 'low'] = 98.
    signal = np.zeros(8, dtype=np.int8)
    signal[0] = 1
    first = run(df, signal, config(), start=0, end=480)['trades'][0]
    df.loc[4:, ['open', 'high', 'low', 'close']] = 1000.
    signal[4:] = -1
    second = run(df, signal, config(), start=0, end=480)['trades'][0]
    assert first == second


def test_signal_at_fill_bar_cannot_change_prior_entry_barrier():
    df = bars(3)
    df.loc[1, 'atr'] = 1000.
    df.loc[1, 'low'] = 98.
    trade, = execute(df)['trades']
    assert trade['exit'] == pytest.approx(98.8)


def test_scanner_gap_past_proposed_target_rejected():
    df = bars(3)
    df.loc[1:, ['open', 'high', 'low', 'close']] = 103.
    result = execute(df, stops=np.array([99., 0., 0.]), targets=np.array([102., 0., 0.]))
    assert result['trades'] == []
    assert result['rejections']['invalid_barriers'] == 1


@pytest.mark.parametrize('funding', [
    [dict(time=90, rate=.001, mark_price=100.), dict(time=80, rate=.001, mark_price=100.)],
    [dict(time=90, rate=.001, mark_price=100.), dict(time=90, rate=.001, mark_price=100.)],
    [dict(time=90.5, rate=.001, mark_price=100.)],
    [dict(time=90, rate=.001)],
])
def test_malformed_funding_order_or_event_refused(funding):
    with pytest.raises(ValueError):
        execute(bars(3), funding=funding)


@pytest.mark.parametrize('mode', ['atr', 'partial', 'original'])
def test_equity_never_marks_beyond_target_after_position_was_exited(mode):
    df = bars(4)
    df.loc[2, 'high'] = 110.
    result = execute(df, settings=config(exit_mode=mode))
    trade, = result['trades']
    assert trade['exit_reason'] in {'target', 'roi'}
    assert max(value for _, value in result['equity']) <= 1000 + trade['net_pnl'] + 1e-8


@pytest.mark.parametrize('side', [1, -1])
def test_partial_settlement_at_close_uses_reduced_quantity(side):
    df = bars(4)
    df.loc[1, 'high' if side == 1 else 'low'] = 100 + side * 1.1
    df.loc[2:, ['open', 'high', 'low', 'close']] = 100 + side * .5
    result = execute(df, side, settings=config(exit_mode='partial'),
                     funding=[dict(time=120, rate=.001, mark_price=100.)])
    trade, = result['trades']
    assert trade['funding_cost'] == pytest.approx(side * trade['qty'] * .5 * .1)


@pytest.mark.parametrize('minutes,price', [(0, 100.86), (6, 100.14), (16, 100.08), (40, 100.)])
def test_original_roi_schedule_every_tier(minutes, price):
    df = bars(minutes + 5)
    index = max(2, minutes + 1)
    df.loc[index:, ['open', 'high', 'low', 'close']] = price
    trade, = execute(df, settings=config(exit_mode='original'))['trades']
    assert trade['exit_reason'] == 'roi'
    assert trade['exit_time'] == index * 60


@pytest.mark.parametrize('side', [1, -1])
def test_partial_terminal_cost_and_funding_independent_accounting(side):
    df = bars(4)
    df.loc[1, 'high' if side == 1 else 'low'] = 100 + side * 1.3
    df.loc[1, 'close'] = 100 + side * .6
    df.loc[2:, ['open', 'high', 'low', 'close']] = 100 + side * .6
    c = replace(config(exit_mode='partial'), fee=.0005, slippage=.0002, spread_bps=2.)
    result = execute(df, side, settings=c,
                     funding=[dict(time=90, rate=.001, mark_price=100.),
                              dict(time=150, rate=-.0005, mark_price=100.)])
    trade, = result['trades']
    initial = 100 * (1 + side * .0003)
    first_mid = initial + side
    last_mid = 100 + side * .6
    weighted_exit = (first_mid + last_mid) / 2 * (1 - side * .0003)
    q = trade['qty']
    fee = q * (initial + weighted_exit) * c.fee
    funding = side * q * (.1 - .025)
    gross = side * q * (weighted_exit - initial)
    assert trade['exit_reason'] == 'terminal'
    assert trade['exit'] == pytest.approx(weighted_exit)
    assert trade['fees'] == pytest.approx(fee)
    assert trade['spread_cost'] == pytest.approx(q * (100 + (first_mid + last_mid)/2) * .0001)
    assert trade['slippage_cost'] == pytest.approx(q * (100 + (first_mid + last_mid)/2) * .0002)
    assert trade['funding_cost'] == pytest.approx(funding)
    assert trade['net_pnl'] == pytest.approx(gross - fee - funding)
    assert result['equity'][-1][1] == pytest.approx(1000 + gross - fee - funding)


@pytest.mark.parametrize('side', [1, -1])
def test_partial_break_even_allocates_post_partial_funding_to_remaining_leg(side):
    df = bars(5)
    df.loc[1, 'high' if side == 1 else 'low'] = 100 + side * 1.1
    df.loc[1, 'close'] = 100 + side * .5
    df.loc[2:, ['open', 'high', 'low', 'close']] = 100 + side * .5
    df.loc[3, 'low' if side == 1 else 'high'] = 100 - side * .1
    result = execute(df, side, settings=config(exit_mode='partial'),
                     funding=[dict(time=150, rate=side * .001, mark_price=100.)])
    trade, = result['trades']
    assert trade['exit_reason'] == 'stop'
    assert trade['net_pnl'] == pytest.approx(trade['qty'] / 2)
    assert trade['exit'] == pytest.approx(100 + side * .55)


def test_increasing_rangeindex_need_not_start_zero():
    df = bars(3)
    df.index = pd.RangeIndex(600, 603)
    trade, = execute(df)['trades']
    assert trade['entry_time'] == 60


@pytest.mark.parametrize('side', [1, -1])
def test_entry_open_crossed_stop_uses_obtainable_adverse_open(side):
    df = bars(3)
    stop = 100 + side * .02
    c = config(slippage=.0002, spread_bps=2., initial_equity=10000.)
    trade, = execute(df, side, settings=c,
                     stops=np.array([stop, 0., 0.]),
                     targets=np.array([100 + side * 2, 0., 0.]))['trades']
    assert trade['exit_reason'] == 'stop'
    assert trade['exit_time'] == 60
    assert trade['exit'] == pytest.approx(100 * (1 - side * .0003))


@pytest.mark.parametrize('impact', [0., .0003])
def test_short_stop_loss_with_actual_exit_fee_never_exceeds_sized_risk(impact):
    df = bars(3)
    df.loc[1, 'high'] = 108.
    c = config(fee=.001, slippage=impact)
    trade, = execute(df, -1, settings=c, stops=np.array([108., 0., 0.]),
                     targets=np.array([90., 0., 0.]))['trades']
    assert trade['exit_reason'] == 'stop'
    assert -trade['net_pnl'] <= trade['initial_risk_usdt'] + 1e-10


def test_execution_refuses_off_grid_minute_opens():
    df = bars(3)
    df['time'] += 1
    with pytest.raises(ValueError, match='1m|aligned|grid'):
        execute(df)


@pytest.mark.parametrize('side', [1, -1])
def test_explicit_source_exit_switch_preserves_external_exit_mode(side):
    df = bars(20)
    df['exit_long'] = df['exit_short'] = 1
    enabled, = execute(df, side, settings=config(max_hold_bars=15, use_source_exits=True))['trades']
    disabled, = execute(df, side, settings=config(max_hold_bars=15, use_source_exits=False))['trades']
    assert (enabled['exit_reason'], enabled['holding_seconds']) == ('exit_signal', 60)
    assert (disabled['exit_reason'], disabled['holding_seconds']) == ('timeout', 900)


@pytest.mark.parametrize('side', [1, -1])
def test_min_rr_actual_next_open_rejects_fixed_plan_gap(side):
    df = bars(4)
    df.loc[1:, ['open', 'high', 'low', 'close']] = 100 + side * .8
    result = execute(df, side, settings=config(min_net_rr=1.),
                     stops=np.array([100 - side * 2, 0., 0., 0.]),
                     targets=np.array([100 + side * 3, 0., 0., 0.]))
    assert result['trades'] == []
    assert result['rejections'] == {'min_net_rr': 1}


@pytest.mark.parametrize('side', [1, -1])
def test_deadfish_requires_joint_flag_min_hold_stall_and_no_strong_trend(side):
    df = bars(10)
    df['deadfish_long'] = df['deadfish_short'] = True
    df['strong_trend_long'] = df['strong_trend_short'] = False
    c = config(exit_mode='trailing', deadfish_enabled=True)
    trade, = execute(df, side, settings=c)['trades']
    assert (trade['exit_reason'], trade['holding_seconds']) == ('deadfish', 300)
    df['strong_trend_long' if side == 1 else 'strong_trend_short'] = True
    assert execute(df, side, settings=c)['trades'][0]['exit_reason'] == 'terminal'
    df['strong_trend_long'] = df['strong_trend_short'] = False
    df['deadfish_long'] = df['deadfish_short'] = False
    assert execute(df, side, settings=c)['trades'][0]['exit_reason'] == 'terminal'
    df['deadfish_long'] = df['deadfish_short'] = True
    df.loc[2:, ['open', 'high', 'low', 'close']] = 100 + side * .5
    assert execute(df, side, settings=c)['trades'][0]['exit_reason'] == 'terminal'


def test_deadfish_requires_all_feature_flags_not_an_extra_ema_reversal():
    df = bars(10)
    with pytest.raises(ValueError, match='deadfish'):
        execute(df, settings=config(deadfish_enabled=True))
    df['deadfish_long'] = df['deadfish_short'] = True
    df['strong_trend_long'] = df['strong_trend_short'] = False
    df['momentum'] = 1
    assert execute(df, settings=config(deadfish_enabled=True))['trades'][0]['exit_reason'] == 'deadfish'


@pytest.mark.parametrize('side', [1, -1])
def test_deadfish_exits_joint_fading_while_ema_direction_still_aligned(side):
    df = bars(10)
    df['deadfish_long'] = df['deadfish_short'] = True
    df['strong_trend_long'] = df['strong_trend_short'] = False
    df['momentum'] = side  # EMA ordering can persist while its strength fades.
    trade, = execute(df, side, settings=config(exit_mode='trailing', deadfish_enabled=True))['trades']
    assert (trade['exit_reason'], trade['holding_seconds']) == ('deadfish', 300)


def test_trailing_atr_separate_from_initial_stop():
    df = bars(4)
    df.loc[1, ['high', 'low', 'close']] = [100.3, 99.9, 100.2]
    df.loc[2, ['open', 'high', 'low', 'close']] = [100.2, 100.2, 99.1, 100.]
    trade, = execute(df, settings=config(exit_mode='trailing', trailing_atr=.5))['trades']
    assert trade['initial_stop'] == pytest.approx(98.8)
    assert trade['exit'] == pytest.approx(99.8)


@pytest.mark.parametrize('options', [dict(trailing_atr=0), dict(trailing_atr=True),
    dict(trailing_atr=float('nan')), dict(min_net_rr=-1), dict(min_net_rr=True),
    dict(deadfish_enabled=1), dict(deadfish_min_hold_bars=0),
    dict(deadfish_stall_risk_fraction=-.1), dict(deadfish_stall_risk_fraction=1.1)])
def test_extended_cost_fields_validated(options):
    with pytest.raises(ValueError):
        config(**options)
