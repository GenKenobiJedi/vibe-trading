"""Contracts for real data collection; fixtures are synthetic unit tests only."""
import unittest
from research_scalping.data import validate_rows, parse_page, expected_minutes


def row(t):
    return dict(time=t, open=10., high=12., low=9., close=11., volume=2.)


class DataTests(unittest.TestCase):
    def test_exact_closed_range(self):
        s = validate_rows([row(60), row(120)], 60, 180)
        self.assertTrue(s['complete'])
        self.assertEqual(s['expected_rows'], 2)
        self.assertEqual(s['missing_rows'], 0)

    def test_missing_is_reported_not_padded(self):
        s = validate_rows([row(60), row(180)], 60, 240)
        self.assertFalse(s['complete'])
        self.assertEqual(s['missing_rows'], 1)
        self.assertEqual(s['gaps'], [[120, 180]])

    def test_milliseconds_are_normalized_and_sorted(self):
        p = {'code': 0, 'data': [{**row(120), 'time': 120000}, {**row(60), 'time': 60000}]}
        a = parse_page(p, 60, 180)
        self.assertEqual([r['time'] for r in a], [60, 120])

    def test_future_or_outside_requested_range_refused(self):
        with self.assertRaises(ValueError):
            parse_page({'code': 0, 'data': [{**row(180), 'time': 180000}]}, 60, 180)

    def test_exchange_error_not_empty_success(self):
        with self.assertRaises(ValueError):
            parse_page({'code': 109400, 'msg': 'unknown', 'data': None}, 60, 180)

    def test_duplicate_refused(self):
        with self.assertRaises(ValueError):
            validate_rows([row(60), row(60)], 60, 180)

    def test_bad_ohlc_refused(self):
        with self.assertRaises(ValueError):
            validate_rows([{**row(60), 'high': 9.}], 60, 120)

    def test_nan_refused(self):
        with self.assertRaises(ValueError):
            validate_rows([{**row(60), 'close': float('nan')}], 60, 120)

    def test_unaligned_time_refused(self):
        with self.assertRaises(ValueError):
            validate_rows([row(61)], 60, 120)

    def test_empty_is_not_zero_trade_evidence(self):
        self.assertFalse(validate_rows([], 60, 180)['complete'])

    def test_expected_minutes(self):
        self.assertEqual(expected_minutes(60, 180), 2)
        with self.assertRaises(ValueError):
            expected_minutes(60, 181)


if __name__ == '__main__':
    unittest.main()


import gzip
import hashlib
import json
import pytest
from research_scalping import data


@pytest.mark.parametrize('cached', [{'code': 109400, 'data': None},
                                  {'code': 0, 'data': [{**row(0), 'time': 0}]}])
def test_resume_refetches_failed_or_incomplete_cache_retaining_original(tmp_path, monkeypatch, cached):
    monkeypatch.setattr(data, 'WARM_START', 0)
    monkeypatch.setattr(data, 'START', 0)
    monkeypatch.setattr(data, 'END', 120)
    monkeypatch.setattr(data.time, 'sleep', lambda _: None)
    folder = tmp_path/'raw'/'SYNTHUSDT'
    folder.mkdir(parents=True)
    raw = json.dumps(cached, separators=(',', ':')).encode()
    original = gzip.compress(raw, mtime=0)
    (folder/'0.json.gz').write_bytes(original)
    (folder/'0.meta.json').write_text(json.dumps({'normalized_payload_sha256': hashlib.sha256(raw).hexdigest()}))
    calls = []
    def get(path, params):
        calls.append(path)
        if 'fundingRate' in path:
            return {'code': 0, 'data': []}, {}
        return {'code': 0, 'data': [{**row(t), 'time': t*1000} for t in (0, 60)]}, {}
    monkeypatch.setattr(data, 'get_public', get)
    result = data.collect_pair('SYNTHUSDT', True, tmp_path)
    assert any('klines' in path for path in calls)
    assert result['coverage']['complete']
    assert any(p.read_bytes() == original for p in folder.glob('*.json.gz') if p.name != '0.json.gz')
    calls.clear()
    assert data.collect_pair('SYNTHUSDT', True, tmp_path)['coverage']['complete']
    assert not any('klines' in path for path in calls)


def test_resume_failed_refetch_retains_original_without_fake_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'WARM_START', 0)
    monkeypatch.setattr(data, 'START', 0)
    monkeypatch.setattr(data, 'END', 120)
    folder = tmp_path/'raw/SYNTHUSDT'
    folder.mkdir(parents=True)
    raw = b'{"code":109400,"data":null}'
    original = gzip.compress(raw, mtime=0)
    (folder/'0.json.gz').write_bytes(original)
    (folder/'0.meta.json').write_text(json.dumps({'normalized_payload_sha256': hashlib.sha256(raw).hexdigest()}))
    calls = []
    def fail(path, params):
        calls.append(path)
        raise RuntimeError('synthetic fetch failure')
    monkeypatch.setattr(data, 'get_public', fail)
    result = data.collect_pair('SYNTHUSDT', True, tmp_path)
    assert any('klines' in path for path in calls)
    assert result['status'] == 'incomplete_or_unlisted'
    assert result['coverage']['rows'] == 0
    assert result['coverage']['missing_rows'] == 2
    assert not (tmp_path/'SYNTHUSDT.parquet').exists()
    assert (folder/'0.retry1.json.gz').read_bytes() == original
