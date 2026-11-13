"""Synthetic, offline contracts; the fake scanner is not financial evidence."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_ict import build_features, signals
from research_ict import features as f


def bars(n=1200, start=17280000, step=60, direction=1):
    close = 100 + direction * np.arange(n) * .001
    return pd.DataFrame(dict(time=start + np.arange(n, dtype=np.int64) * step,
                             open=close - direction * .0005, high=close + .01,
                             low=close - .01, close=close, volume=np.full(n, 10.)))


@pytest.fixture
def scanner(tmp_path):
    path = tmp_path / 'synthetic_ict.py'
    path.write_text('''import numpy as np
def ict_scan(open_, high, low, close, direction="LONG", require_killzone=False, timestamps=None):
    assert len(close) == len(timestamps) == 200
    assert np.all(np.diff(timestamps) > 0)
    return {"score": 8, "details": {"fvgs": [], "obs": [], "structure": {"trend": "bullish"}}}
''')
    return path


def warmup(start=17280000, direction=1):
    return {key: bars(210, start - 210 * seconds, seconds, direction)
            for key, seconds in [('1h', 3600), ('4h', 14400)]}


def test_preserves_input_and_scanner_provenance(scanner):
    source = bars()
    before = source.copy(deep=True)
    result = build_features(source, ict_module_path=scanner, htf_warmup=warmup())
    pd.testing.assert_frame_equal(source, before)
    pd.testing.assert_frame_equal(result[source.columns], source)
    assert result.index.equals(source.index)
    assert result.attrs['provenance']['ict_module_sha256']
    assert result.attrs['counts']['ict_scan_calls'] < len(source) / 10
    assert result.momentum.dtype == np.int8
    assert result.trend1h.iloc[-1] == result.trend4h.iloc[-1] == 1
    assert result.atr.iloc[998] != result.atr.iloc[999]


@pytest.mark.parametrize('direction', [1, -1])
def test_trend_and_insufficient_warmup(scanner, direction):
    result = build_features(bars(direction=direction), ict_module_path=scanner,
                            htf_warmup=warmup(direction=direction))
    assert result.trend1h.iloc[-1] == result.trend4h.iloc[-1] == direction
    assert result.trend5.iloc[998] == 0
    assert result.trend5.iloc[999] == direction
    cold = build_features(bars(599), ict_module_path=scanner)
    for variant in ['baseline', 'sweep_ote', 'pullback', 'sweep_ote_pullback']:
        assert not signals(cold, variant).any()


@pytest.mark.parametrize('broken', ['gap', 'duplicate', 'float_time', 'off_grid', 'negative_time',
                                    'nan', 'infinity', 'negative_volume', 'bad_ohlc', 'index', 'empty'])
def test_reject_malformed(scanner, broken):
    source = bars(20)
    if broken == 'gap':
        source.loc[5:, 'time'] += 60
    if broken == 'duplicate':
        source.loc[5, 'time'] = source.loc[4, 'time']
    if broken == 'float_time':
        source['time'] = source.time.astype(float)
    if broken == 'off_grid':
        source['time'] += 1
    if broken == 'negative_time':
        source['time'] -= 20000000
    if broken == 'nan':
        source.loc[3, 'close'] = np.nan
    if broken == 'infinity':
        source.loc[3, 'high'] = np.inf
    if broken == 'negative_volume':
        source.loc[3, 'volume'] = -1
    if broken == 'bad_ohlc':
        source.loc[3, 'low'] = 200
    if broken == 'index':
        source.index = pd.Index(np.arange(20) + 1)
    if broken == 'empty':
        source = source.iloc[:0]
    with pytest.raises(ValueError):
        build_features(source, ict_module_path=scanner)


def test_exact_close_alignment_and_incomplete_groups():
    source = bars(241)
    hourly = f._aggregate(source, 3600)
    assert len(hourly) == 4
    aligned = f._align(source.time.to_numpy() + 60, hourly.time.to_numpy() + 3600,
                       np.arange(4) + 10)
    assert np.isnan(aligned[58])
    assert aligned[59] == 10
    assert aligned[60] == 10
    assert aligned[118] == 10
    assert aligned[119] == 11
    assert len(f._aggregate(source.iloc[1:].reset_index(drop=True), 3600)) == 3


def test_warmup_rejects_overlap_gap_unknown_key(scanner):
    for bad in [dict(warmup(), other=bars(1)), {'1h': bars(210, step=3600)},
                {'1h': warmup()['1h'].iloc[:-1].reset_index(drop=True)}]:
        with pytest.raises(ValueError):
            build_features(bars(), ict_module_path=scanner, htf_warmup=bad)


def test_reject_legacy_scanner(tmp_path):
    path = tmp_path / 'legacy.py'
    path.write_text('def ict_scan(open_, high, low, close): return {}')
    with pytest.raises(ValueError, match='timestamps'):
        build_features(bars(), ict_module_path=path)


def test_pivot_confirmation_delay_and_ties():
    source = bars(12)
    source['high'] = [101, 102, 103, 110, 105, 104, 103, 102, 101, 101, 101, 101]
    highs, _ = f._pivots(source, 3)
    assert np.isnan(highs[:6]).all()
    assert highs[6] == 110
    assert np.isnan(highs[7:]).all()
    highs5, _ = f._pivots(source, 2)
    assert highs5[5] == 110
    source.loc[4, 'high'] = 110
    assert np.isnan(f._pivots(source, 3)[0]).all()


def sequence_fixture(monkeypatch, sign=1, ote_close=102.5, zone_at=4, tap_at=6, n=12):
    source = bars(n)
    source.loc[:, ['open', 'high', 'low', 'close']] = [106., 107., 105., 106.]
    source.loc[2, ['open', 'high', 'low', 'close']] = [101., 102., 100., 101.5]
    source.loc[3, ['open', 'high', 'low', 'close']] = [106., 110., 105., 109.]
    source.loc[4, ['open', 'high', 'low', 'close']] = [108., 109., 107., 108.]
    source.loc[5, ['open', 'high', 'low', 'close']] = [103., 103.1, 102.3, ote_close]
    source.loc[tap_at, ['open', 'high', 'low', 'close']] = [102.2, 104., 102.15, 103.5]
    hp, lp = np.full(n, np.nan), np.full(n, np.nan)
    hp[0], lp[0] = 108, 101
    formations = {}
    formations[zone_at] = [(1, 102.1, 102.2, zone_at, zone_at - 1, 'ob')]
    if sign == -1:
        original = source[['open', 'high', 'low', 'close']].copy()
        source['open'], source['close'] = 220 - original.open, 220 - original.close
        source['high'], source['low'] = 220 - original.low, 220 - original.high
        hp, lp = 220 - lp, 220 - hp
        formations[zone_at] = [(-1, 117.8, 117.9, zone_at, zone_at - 1, 'ob')]
    monkeypatch.setattr(f, '_pivots', lambda df, window: (hp, lp))
    monkeypatch.setattr(f, '_formations', lambda df: formations)
    return source


@pytest.mark.parametrize('sign', [1, -1])
def test_sequential_setup_each_stage_and_frozen_anchors(monkeypatch, sign):
    source = sequence_fixture(monkeypatch, sign)
    result = f._sequence(source, np.full(len(source), sign))
    side = 'long' if sign == 1 else 'short'
    assert not result['sweep_' + side][:2].any()
    assert result['sweep_' + side][2]
    assert result['structure_' + side][3]
    assert not result['ote_' + side][:5].any()
    assert result['ote_' + side][5]
    assert not result['fresh_zone_' + side][:6].any()
    assert result['fresh_zone_' + side][6]
    assert result['ote_lower_' + side][6] == pytest.approx(102.1 if sign == 1 else 116.2)
    assert result['ote_upper_' + side][6] == pytest.approx(103.8 if sign == 1 else 117.9)
    assert result['structural_stop_' + side][6] == (100 if sign == 1 else 120)
    assert not result['fresh_zone_' + side][7:].any()


@pytest.mark.parametrize('sign', [1, -1])
@pytest.mark.parametrize('endpoint', [102.1, 103.8, 102.099, 103.801])
def test_ote_endpoints(monkeypatch, sign, endpoint):
    source = sequence_fixture(monkeypatch, sign, ote_close=endpoint)
    # Keep endpoint candles geometrically valid.
    source.loc[5, ['high', 'low']] = ([104, 102] if sign == 1 else [118, 116])
    result = f._sequence(source, np.full(len(source), sign))
    assert bool(result['ote_' + ('long' if sign == 1 else 'short')][5]) == (102.1 <= endpoint <= 103.8)


@pytest.mark.parametrize('failure', ['same_ote_tap', 'zone_unknown', 'invalidated', 'expired', 'trend_lost', 'wick_break'])
@pytest.mark.parametrize('sign', [1, -1])
def test_sequence_failure_gates(monkeypatch, sign, failure):
    n = 70 if failure == 'expired' else 12
    tap = 64 if failure == 'expired' else 6
    source = sequence_fixture(monkeypatch, sign, zone_at=6 if failure == 'zone_unknown' else 4,
                              tap_at=tap, n=n)
    side = 'long' if sign == 1 else 'short'
    if failure == 'same_ote_tap':
        source.loc[5, 'low' if sign == 1 else 'high'] = 101.8 if sign == 1 else 118.2
    if failure == 'invalidated':
        source.loc[5, 'low' if sign == 1 else 'high'] = 101 if sign == 1 else 119
    if failure == 'wick_break':
        source.loc[3, 'close'] = 107 if sign == 1 else 113
    direction = np.full(n, sign)
    if failure == 'trend_lost':
        direction[4] = 0
    result = f._sequence(source, direction)
    assert not result['fresh_zone_' + side].any()


@pytest.mark.parametrize('sign', [1, -1])
def test_baseline_zone_reaction_and_retirement(sign):
    source = bars(6)
    source.loc[:, ['open', 'high', 'low', 'close']] = [102., 104., 101.8, 103.]
    source['ready'] = True
    for name in ['trend5', 'trend1h', 'trend4h']:
        source[name] = sign
    for side, s in f.SIDES:
        source['reaction_' + side] = s == sign
        source['ict_score1h_' + side] = 6
        source['ict_score4h_' + side] = 5
    if sign == -1:
        source['open'], source['close'], source['high'], source['low'] = 118., 117., 118.2, 116.
    bottom, top = (101.7, 101.9) if sign == 1 else (118.1, 118.3)
    zones = [[], []]
    zones[0 if sign == 1 else 1] = [(bottom, top, int(source.time.iloc[0]) - 3600, 'fvg')]
    known = np.array([source.time.iloc[0]])
    contexts = {'1h': (known, [zones]), '4h': (known, [[[], []]])}
    result = f._baseline(source, contexts)
    side = 'long' if sign == 1 else 'short'
    assert result['baseline_' + side][0]
    assert not result['baseline_' + side][1:].any()
    source.loc[0, 'volume'] = 0
    assert not f._baseline(source, contexts)['baseline_' + side].any()
    source.loc[0, 'volume'] = 10
    source.loc[0, 'ict_score4h_' + side] = 4
    assert not f._baseline(source, contexts)['baseline_' + side].any()


def test_ha_recursive_and_prefix():
    source = bars(90)
    source.loc[2, ['open', 'high', 'low', 'close']] = [99, 102, 98, 101]
    result = f._minute_indicators(source)
    assert result['ha_open'][0] == pytest.approx((source.open.iloc[0] + source.close.iloc[0]) / 2)
    for i in range(1, len(source)):
        assert result['ha_open'][i] == pytest.approx((result['ha_open'][i-1] + result['ha_close'][i-1]) / 2)
    for key, value in f._minute_indicators(source.iloc[:50]).items():
        np.testing.assert_allclose(value, result[key][:50], equal_nan=True)


@pytest.mark.parametrize('sign', [1, -1])
@pytest.mark.parametrize('missing', [None, 'volume', 'width', 'momentum', 'strong'])
def test_deadfish_requires_joint_weakening_and_guard(sign, missing):
    source = pd.DataFrame(dict(bb20_width=np.r_[np.ones(60), [.7, .6, .5, .4, .3]],
                               volume_ratio12_24=np.full(65, .5),
                               momentum_delta=np.full(65, -sign * .01),
                               strong_trend_long=False, strong_trend_short=False))
    if missing == 'volume':
        source['volume_ratio12_24'] = .9
    if missing == 'width':
        source['bb20_width'] = 1.
    if missing == 'momentum':
        source.loc[64, 'momentum_delta'] *= -1
    if missing == 'strong':
        source['strong_trend_' + ('long' if sign == 1 else 'short')] = True
    result = f._deadfish(source)
    assert bool(result['deadfish_' + ('long' if sign == 1 else 'short')][-1]) == (missing is None)


def signal_frame(sign):
    frame = bars(5)
    frame['ready'] = True
    for name in ['trend5', 'trend1h', 'trend4h']:
        frame[name] = sign
    frame['volume'] = [12, 11.99, 12, 12, 0]
    frame['volume_prior20'] = 10.
    for side, s in f.SIDES:
        for name in ['baseline', 'sweep', 'ote', 'fresh_zone', 'reaction', 'pullback']:
            frame[name + '_' + side] = s == sign
    return frame


@pytest.mark.parametrize('variant', ['baseline', 'sweep_ote', 'pullback', 'sweep_ote_pullback'])
@pytest.mark.parametrize('sign', [1, -1])
def test_variant_signal_gates_and_volume(variant, sign):
    frame = signal_frame(sign)
    frame.loc[2, 'trend4h'] = -sign
    frame.loc[3, 'pullback_' + ('long' if sign == 1 else 'short')] = False
    before = frame.copy(deep=True)
    result = signals(frame, variant)
    expected = [sign, 0 if 'sweep' in variant else sign, 0,
                0 if 'pullback' in variant else sign, 0]
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.int8
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize('params', [{'volume_ratio': .99}, {'volume_ratio': 3.01}, {'volume_ratio': np.nan},
                                  {'volume_ratio': True}, {'volume_ratio': '1.2'}, {'volume_ratio': None},
                                  {'foo': 1}, [], {'volume_ratio': np.bool_(True)}, {'volume_ratio': 1.2+0j}])
def test_signal_params_reject(params):
    with pytest.raises(ValueError): signals(signal_frame(1), 'baseline', params)


@pytest.mark.parametrize('seconds', [3600, 14400])
def test_scanner_uses_actual_htf_evaluation_close(seconds):
    htf = bars(201, start=0, step=seconds)
    seen = []
    def scanner(*prices, timestamps, **kwargs):
        seen.append(timestamps.copy())
        return dict(score=0, details=dict(fvgs=[], obs=[]))
    f._scan_context(htf, scanner, seconds)
    assert len(seen) == 4
    assert seen[0][-1] == 200 * seconds
    assert seen[-1][-1] == 201 * seconds


def test_5m_direction_does_not_add_unrequested_price_or_slope_gate():
    source = bars(230, step=300)
    source.loc[229, ['open', 'low', 'close']] = [100.20, 100.19, 100.20]
    direction, fast, slow, _ = f._trend(source, 9, 21, 200, price_slope=False)
    assert source.close.iloc[-1] < fast[-1]
    assert fast[-1] > slow[-1]
    assert direction[-1] == 1


@pytest.mark.parametrize('sign', [1, -1])
def test_pullback_uses_real_recursive_ha_bb40_rsi_both_sides(sign):
    close = np.r_[100 + np.sin(np.arange(70)) * .1, 100 - np.arange(1, 6) * .2,
                  99 + np.arange(1, 7) / 6]
    if sign == -1:
        close = 200 - close
    opening = np.r_[close[0], close[:-1]]
    source = bars(len(close))
    source['open'], source['close'] = opening, close
    source['high'], source['low'] = np.maximum(opening, close) + .01, np.minimum(opening, close) - .01
    result = f._minute_indicators(source)
    side = 'long' if sign == 1 else 'short'
    assert result['pullback_' + side][77]
    assert not result['pullback_' + side][:39].any()
    assert 30 <= result['rsi14'][77] <= 55 if sign == 1 else 45 <= result['rsi14'][77] <= 70


def test_future_mutation_and_every_prefix_boundary(scanner):
    source = bars(1320)
    prelude = warmup()
    full = build_features(source, ict_module_path=scanner, htf_warmup=prelude)
    for cut in [599, 600, 999, 1000, 1019, 1020, 1199, 1200, 1201]:
        prefix = build_features(source.iloc[:cut].copy(), ict_module_path=scanner, htf_warmup=prelude)
        pd.testing.assert_frame_equal(prefix, full.iloc[:cut], check_flags=False)
        changed = source.copy()
        changed.loc[cut:, ['open', 'high', 'low', 'close']] *= 1.1
        changed.loc[cut:, 'volume'] *= 20
        mutated = build_features(changed, ict_module_path=scanner, htf_warmup=prelude)
        pd.testing.assert_frame_equal(mutated.iloc[:cut], full.iloc[:cut], check_flags=False)
        for variant in ['baseline', 'sweep_ote', 'pullback', 'sweep_ote_pullback']:
            np.testing.assert_array_equal(signals(prefix, variant), signals(full, variant)[:cut])


def raw_sequence(n=90):
    source = bars(n)
    source.loc[:, ['open', 'high', 'low', 'close']] = [106., 107., 105., 106.]
    changes = {5: [106, 108, 105, 106], 12: [104, 105, 101, 103],
               13: [104, 105, 103, 104], 14: [105, 106, 104, 105],
               16: [102, 103, 100, 101.5], 17: [106, 110, 105, 109],
               18: [108, 109, 107, 108], 19: [103, 104, 102, 102.5],
               20: [103, 103.2, 101.5, 102.5], 21: [102.6, 104.2, 102.5, 103.3],
               22: [103, 105, 103, 103.5]}
    for i, values in changes.items():
        source.loc[i, ['open', 'high', 'low', 'close']] = values
    return source


def mirror(source):
    result = source.copy()
    result['open'], result['close'] = 220 - source.open, 220 - source.close
    result['high'], result['low'] = 220 - source.low, 220 - source.high
    return result


@pytest.mark.parametrize('sign', [1, -1])
def test_raw_ohlcv_pipeline_emits_both_sides_and_is_prefix_invariant(scanner, sign):
    source = bars(1200)
    raw = raw_sequence()
    source.loc[1000:1089, ['open', 'high', 'low', 'close']] = raw[['open', 'high', 'low', 'close']].to_numpy()
    source.loc[1022, 'volume'] = 12
    prelude = warmup()
    if sign == -1:
        source = mirror(source)
        prelude = {key: mirror(value) for key, value in prelude.items()}
    full = build_features(source, ict_module_path=scanner, htf_warmup=prelude)
    side = 'long' if sign == 1 else 'short'
    assert full['fresh_zone_' + side].iloc[1022]
    assert signals(full, 'sweep_ote')[1022] == sign
    for cut in [1009, 1016, 1017, 1018, 1020, 1022, 1023]:
        prefix = build_features(source.iloc[:cut], ict_module_path=scanner, htf_warmup=prelude)
        pd.testing.assert_frame_equal(prefix, full.iloc[:cut])
        mutated = source.copy()
        mutated.loc[cut:, ['open', 'high', 'low', 'close']] *= .9
        mutated.loc[cut:, 'volume'] *= 100
        result = build_features(mutated, ict_module_path=scanner, htf_warmup=prelude)
        pd.testing.assert_frame_equal(result.iloc[:cut], full.iloc[:cut])


@pytest.mark.parametrize('sign', [1, -1])
def test_local_zone_age_30_and_far_boundary(monkeypatch, sign):
    source = sequence_fixture(monkeypatch, sign, zone_at=4, tap_at=35, n=40)
    side = 'long' if sign == 1 else 'short'
    result = f._sequence(source, np.full(len(source), sign))
    assert not result['fresh_zone_' + side].any()
    source = sequence_fixture(monkeypatch, sign, zone_at=4)
    source.loc[6, 'low' if sign == 1 else 'high'] = 101.69 if sign == 1 else 118.31
    assert not f._sequence(source, np.full(len(source), sign))['fresh_zone_' + side].any()


@pytest.mark.parametrize('sign', [1, -1])
def test_later_peak_does_not_move_frozen_ote(monkeypatch, sign):
    source = sequence_fixture(monkeypatch, sign)
    source.loc[4, 'high' if sign == 1 else 'low'] = 115 if sign == 1 else 105
    result = f._sequence(source, np.full(len(source), sign))
    side = 'long' if sign == 1 else 'short'
    assert result['ote_lower_' + side][6] == pytest.approx(102.1 if sign == 1 else 116.2)
    assert result['fresh_zone_' + side][6]


def test_real_corrected_scanner_parity_on_synthetic_prefixes():
    path = Path(__file__).resolve().parents[4] / 'neuro' / 'ict_analysis.py'
    if not path.exists():
        pytest.skip('Parent integration supplies corrected local module')
    scanner, _ = f._load_scanner(path)
    for seconds in [3600, 14400]:
        source = bars(220)
        source.loc[180:, ['open', 'high', 'low', 'close']] = raw_sequence().iloc[:40][
            ['open', 'high', 'low', 'close']].to_numpy()
        source['time'] = np.arange(len(source), dtype=np.int64) * seconds
        scores, zones, count = f._scan_context(source, scanner, seconds)
        assert count == 42
        assert np.nanmax(scores) > 0
        assert any(snapshot[0] or snapshot[1] for snapshot in zones)
        for end in [199, 200, 201, 202, 219]:
            prefix = source.iloc[end - 199:end + 1]
            for column, (side, _) in enumerate(f.SIDES):
                actual = scanner(*(prefix[name].to_numpy(copy=True) for name in f.FIELDS[1:5]),
                                 direction=side.upper(), timestamps=prefix.time.to_numpy() + seconds,
                                 require_killzone=False)
                assert scores[end, column] == actual['score']


def test_signals_reject_unknown_variant():
    with pytest.raises(ValueError, match='Unknown variant'):
        signals(signal_frame(1), 'missing')


def test_explicit_none_warmup_is_malformed(scanner):
    with pytest.raises(ValueError):
        build_features(bars(), ict_module_path=scanner, htf_warmup={'1h': None})


def test_warmup_immutability_and_no_import_or_generation_side_effects(scanner, monkeypatch):
    import importlib
    import socket
    import subprocess

    prelude = warmup()
    before = {key: value.copy(deep=True) for key, value in prelude.items()}
    def forbidden(*args, **kwargs):
        raise AssertionError('Import/generation attempted an external side effect')
    with monkeypatch.context() as patch:
        patch.setattr(socket, 'socket', forbidden)
        patch.setattr(subprocess, 'Popen', forbidden)
        patch.setattr(Path, 'write_text', forbidden)
        patch.setattr(Path, 'write_bytes', forbidden)
        importlib.reload(f)
        frame = f.build_features(bars(), ict_module_path=scanner, htf_warmup=prelude)
        for variant in ['baseline', 'sweep_ote', 'pullback', 'sweep_ote_pullback']:
            f.signals(frame, variant)
    for key in prelude:
        pd.testing.assert_frame_equal(prelude[key], before[key])


def test_scanner_receives_copies_not_source_views(tmp_path):
    path = tmp_path / 'mutating_synthetic.py'
    path.write_text('''def ict_scan(open_, high, low, close, direction="LONG", require_killzone=False, timestamps=None):
    open_[:] = high[:] = low[:] = close[:] = 1
    timestamps[:] = 0
    return {"score": 0, "details": {"fvgs": [], "obs": []}}
''')
    source, prelude = bars(), warmup()
    expected = source.copy(deep=True)
    before = {key: value.copy(deep=True) for key, value in prelude.items()}
    result = build_features(source, ict_module_path=path, htf_warmup=prelude)
    pd.testing.assert_frame_equal(source, expected)
    pd.testing.assert_frame_equal(result[source.columns], expected)
    for key in prelude:
        pd.testing.assert_frame_equal(prelude[key], before[key])


def test_preserves_any_increasing_range_index(scanner):
    source = bars()
    source.index = pd.RangeIndex(7, 7 + len(source) * 2, 2)
    result = build_features(source, ict_module_path=scanner, htf_warmup=warmup())
    assert result.index.equals(source.index)
    pd.testing.assert_frame_equal(result[source.columns], source)
