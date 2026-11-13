"""Arithmetic and mark-to-market drawdown tests for research summaries."""
import pytest

from research_scalping.metrics import summarize


def trade(net, hold=60., risk=10., reason='stop'):
    return dict(net_pnl=net, holding_seconds=hold, net_r=net/risk,
                fees=1., spread_cost=.1, slippage_cost=.2, funding_cost=-.3,
                exit_reason=reason)


def test_no_trades_has_null_wr_pf_and_zero_pnl_dd():
    result = summarize([], [[0, 1000.]])
    assert result['N'] == result['trades'] == 0
    assert result['net_pnl'] == result['return_pct'] == result['max_drawdown_pct'] == 0
    assert result['win_rate'] is result['profit_factor'] is None
    assert result['profit_factor_status'] == 'no_trades'
    assert result['exit_reasons'] == {}


def test_metrics_exact_arithmetic_and_intratrade_equity_drawdown():
    trades = [trade(20, 60, reason='target'), trade(-10, 180), trade(0, 120)]
    equity = [[0, 1000.], [60, 1020.], [90, 900.], [180, 1010.]]
    result = summarize(trades, equity)
    assert result['N'] == result['trades'] == 3
    assert result['win_rate'] == pytest.approx(1/3)
    assert result['net_pnl'] == 10
    assert result['return_pct'] == 1
    assert result['profit_factor'] == 2
    assert result['profit_factor_status'] == 'ok'
    assert result['max_drawdown_pct'] == pytest.approx(120/1020*100)
    assert result['avg_hold_seconds'] == result['median_hold_seconds'] == 120
    assert result['avg_net_r'] == pytest.approx(1/3)
    assert result['fees'] == 3
    assert result['spread_cost'] == pytest.approx(.3)
    assert result['slippage_cost'] == pytest.approx(.6)
    assert result['funding_cost'] == pytest.approx(-.9)
    assert result['exit_reasons'] == {'target': 1, 'stop': 2}


@pytest.mark.parametrize('nets,status', [([1., 2.], 'no_losses'), ([0., 0.], 'no_losses')])
def test_pf_no_loss_denominator_is_null_with_explicit_status(nets, status):
    result = summarize([trade(n) for n in nets], [[0, 1000], [60, 1000+sum(nets)]])
    assert result['profit_factor'] is None
    assert result['profit_factor_status'] == status


def test_all_losers_pf_zero_and_initial_capital_in_drawdown():
    result = summarize([trade(-100)], [[60, 900.]])
    assert result['profit_factor'] == 0
    assert result['win_rate'] == 0
    assert result['max_drawdown_pct'] == 10


def test_wrong_equity_subset_or_failed_reconciliation_is_rejected():
    with pytest.raises(ValueError, match='reconcil'):
        summarize([trade(20)], [[0, 1000.], [60, 1030.]])


@pytest.mark.parametrize('equity', [[[60, 1000], [0, 1000]], [[0, float('nan')]]])
def test_bad_equity_refused(equity):
    with pytest.raises(ValueError):
        summarize([], equity)


def test_negative_equity_drawdown_can_exceed_one_hundred_percent():
    result = summarize([trade(-1100)], [[0, 1000], [60, -100]])
    assert result['max_drawdown_pct'] == pytest.approx(110)


@pytest.mark.parametrize('initial', [0., -1., float('inf')])
def test_invalid_initial_equity_refused(initial):
    with pytest.raises(ValueError):
        summarize([], [], initial)


@pytest.mark.parametrize('base', [0., -100.])
def test_explicit_insolvent_reporting_keeps_dollars_and_null_equity_percentages(base):
    result = summarize([trade(-10)], [[0, base], [60, base-10]], base,
                       allow_nonpositive_initial=True)
    assert result['net_pnl'] == -10
    assert result['fees'] == 1
    assert result['max_drawdown_usdt'] == 10
    assert result['return_pct'] is result['max_drawdown_pct'] is None
    assert result['equity_percentage_status'] == 'nonpositive_initial_equity'
