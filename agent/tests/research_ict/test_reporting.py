"""Synthetic accounting and immutable artifact tests; no historical results."""
import datetime as dt
import gzip
import json

import pytest

from research_ict import reporting as r


def stamp(text):
    return int(dt.datetime.fromisoformat(text).replace(tzinfo=dt.timezone.utc).timestamp())


def trade(pnl, when, side='LONG'):
    return dict(net_pnl=pnl, exit_time=when, entry_time=when-60, side=side,
                net_r=pnl/10, holding_seconds=60, exit_reason='terminal',
                fees=1., slippage_cost=.2, spread_cost=.1, funding_cost=.3)


def test_seals_canonical_roundtrip_tuples_and_tamper(tmp_path):
    path = tmp_path/'seal.json'
    r.seal(path, {'bounds': (1, 2)})
    assert r.verify_seal(path) == {'bounds': [1, 2]}
    assert r.digest({'a': (1, 2)}) == r.digest({'a': [1, 2]})
    with pytest.raises(FileExistsError):
        r.seal(path, {})
    raw = json.loads(path.read_text())
    raw['payload']['bounds'][0] = 3
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='seal'):
        r.verify_seal(path)


def test_month_mtm_not_exit_attribution_and_direction_dd_realized_only():
    start, boundary, end = map(stamp, ('2026-06-30', '2026-07-01', '2026-07-02'))
    ts = [trade(20., end)]
    result = dict(trades=ts, equity=[[start,1000.], [boundary,1010.], [end,1020.]])
    rows = r.groups(result, dict(window_start=start, window_end=end))
    june = next(x for x in rows if x['month']=='2026-06' and x['direction']=='BOTH')
    july = next(x for x in rows if x['month']=='2026-07' and x['direction']=='BOTH')
    assert (june['net_pnl'], june['mtm_net_pnl']) == (0., 10.)
    assert (july['net_pnl'], july['mtm_net_pnl']) == (20., 10.)
    side = next(x for x in rows if x['month']=='ALL' and x['direction']=='LONG')
    assert side['max_drawdown_pct'] is None
    assert side['realized_drawdown_usdt'] == 0.
    assert 'realized' in side['drawdown_basis']


def test_month_after_bankruptcy_keeps_dollar_drawdown_and_null_percentage():
    start, boundary, end = map(stamp, ('2026-06-30', '2026-07-01', '2026-07-02'))
    ts = [trade(-1100., boundary-60), trade(-50., end)]
    result = dict(trades=ts, equity=[[start,1000.], [boundary,-100.], [boundary+60,-150.], [end,-150.]])
    july = next(x for x in r.groups(result, dict(window_start=start, window_end=end))
                if x['month']=='2026-07' and x['direction']=='BOTH')
    assert july['return_pct'] is None
    assert july['max_drawdown_pct'] is None
    assert july['max_drawdown_usdt'] == 50.
    assert july['mtm_net_pnl'] == -50.


def test_gzip_and_csv_artifacts_are_real_atomic_files(tmp_path):
    path = tmp_path/'ledger.json.gz'
    r.atomic_json(path, {'equity': [[0, 1000.]]})
    assert json.loads(gzip.decompress(path.read_bytes())) == {'equity': [[0, 1000.]]}
    r.atomic_csv(tmp_path/'groups.csv', [{'N':0, 'exit_reasons': {}, 'win_rate':None}])
    assert 'N' in (tmp_path/'groups.csv').read_text()
    assert not list(tmp_path.glob('*.pending'))


def test_no_trade_and_unavailable_matrix_are_not_success():
    metadata = dict(window_start=stamp('2026-06-13'), window_end=stamp('2026-09-13'))
    rows = r.groups(dict(trades=[], equity=[[metadata['window_start'],1000.],
                                          [metadata['window_end'],1000.]]), metadata)
    assert all(x['win_rate'] is None and x['profit_factor'] is None for x in rows)
    unavailable = r.unavailable(metadata, 'MATIC unavailable')
    assert len(unavailable) == len(rows)
    assert all(x['status']=='NOT_EVALUABLE' and x['N'] is None for x in unavailable)


def test_pool_uses_close_equity_and_exact_duration_histogram(tmp_path):
    a = dict(trades=[trade(10.,60),trade(-20.,120)],equity=[[0,1000.],[60,1010.],[120,990.]],
             rejections={'min_net_rr':2})
    b = dict(trades=[trade(20.,120,'SHORT')],equity=[[0,1000.],[60,1000.],[120,1020.]],
             rejections={'min_net_rr':3})
    a['trades'][1]['holding_seconds']=180
    b['trades'][0]['holding_seconds']=120
    paths = [tmp_path/'a.json.gz',tmp_path/'b.json.gz']
    for path,result in zip(paths,[a,b]): r.atomic_json(path,result)
    summary = r.combine(paths)
    assert summary['N']==3 and summary['net_pnl']==10.
    assert summary['win_rate']==pytest.approx(2/3)
    assert summary['profit_factor']==pytest.approx(1.5)
    assert summary['median_hold_seconds']==120.
    assert summary['avg_hold_seconds']==120.
    assert summary['max_drawdown_usdt']==0.
    assert summary['rejections']['min_net_rr']==5
    assert summary['coin_sleeves']==2
