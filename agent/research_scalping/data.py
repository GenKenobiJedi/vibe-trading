"""Public BingX GET-only collector with resumable evidence and strict coverage.

No broker, live scanner, environment loader or account access is imported.
Raw daily API responses are retained beside normalized parquet data. An absent
symbol or minute is never replaced by a different venue, contract or fake bar.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path
import threading
import time
import urllib.parse
import urllib.request

BASE = 'https://open-api.bingx.com'
START = int(dt.datetime(2026, 6, 13, tzinfo=dt.timezone.utc).timestamp())
END = int(dt.datetime(2026, 11, 13, tzinfo=dt.timezone.utc).timestamp())
WARM_START = START - 86400
FIELDS = ('open', 'high', 'low', 'close', 'volume')


def expected_minutes(start, end):
    if type(start) is not int or type(end) is not int or start % 60 or end % 60 or end <= start:
        raise ValueError('Minute-aligned half-open integer range required')
    return (end-start)//60


def validate_rows(rows, start, end):
    expected = expected_minutes(start, end)
    previous = None
    gaps = []
    cursor = start
    for row in rows:
        stamp = row['time']
        if type(stamp) is not int or stamp % 60 or not start <= stamp < end:
            raise ValueError('Out-of-range/off-grid candle')
        if previous is not None and stamp <= previous:
            raise ValueError('Duplicate or unordered candle')
        vals = [float(row[k]) for k in FIELDS]
        o, h, l, c, v = vals
        if not all(math.isfinite(x) for x in vals) or min(o,h,l,c) <= 0 or v < 0 or not l <= min(o,c) <= max(o,c) <= h:
            raise ValueError('Invalid OHLCV candle')
        if stamp > cursor:
            gaps.append([cursor, stamp])
        cursor, previous = stamp+60, stamp
    if cursor < end:
        gaps.append([cursor, end])
    return {'rows': len(rows), 'expected_rows': expected, 'missing_rows': expected-len(rows),
            'gaps': gaps, 'complete': len(rows) == expected and not gaps,
            'start': rows[0]['time'] if rows else None,
            'end_exclusive': rows[-1]['time']+60 if rows else None}


def parse_page(payload, start, end):
    if payload.get('code') != 0:
        raise ValueError('BingX API code '+str(payload.get('code')))
    raw = payload.get('data') or []
    if not isinstance(raw, list):
        raise ValueError('Candle list required')
    rows = [{'time': int(r['time'])//1000, **{k:float(r[k]) for k in FIELDS}} for r in raw]
    if any(int(r['time']) % 1000 for r in raw):
        raise ValueError('Subsecond candle time')
    rows.sort(key=lambda r:r['time'])
    validate_rows(rows, start, end)
    return rows


def get_public(path, params):
    url = BASE+path+'?'+urllib.parse.urlencode(params)
    # Only idempotent public GETs; bounded retries never apply to orders/model calls.
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent':'neuro-issue14-research/1.0'})
            with urllib.request.urlopen(req, timeout=25) as response:
                raw = response.read()
            return json.loads(raw), {'url':url, 'response_sha256':hashlib.sha256(raw).hexdigest(),
                    'retrieved_at':dt.datetime.now(dt.timezone.utc).isoformat()}
        except Exception as exc:
            last = exc
            if attempt < 2:
                time.sleep(2**attempt)
    raise RuntimeError('Public GET failed: '+type(last).__name__)


def _save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.pending')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False))
    temporary.replace(path)


def collect_pair(symbol, listed, output):
    import pandas as pd
    pair = symbol[:-4]+'-USDT'
    folder = output/'raw'/symbol
    folder.mkdir(parents=True, exist_ok=True)
    result = {'symbol':symbol, 'venue_symbol':pair, 'listed_at_collection':listed,
              'source':BASE+'/openApi/swap/v2/quote/klines', 'pages':[], 'errors':[]}
    all_rows = []
    # An unlisted configured contract is probed explicitly, never renamed.
    windows = range(WARM_START, END, 86400) if listed else (START,)
    for n, start in enumerate(windows):
        end = min(start+86400, END)
        page_path = folder/(str(start)+'.json.gz')
        meta_path = folder/(str(start)+'.meta.json')
        try:
            reusable = False
            if page_path.exists() and meta_path.exists():
                try:
                    raw = gzip.decompress(page_path.read_bytes())
                    payload = json.loads(raw)
                    meta = json.loads(meta_path.read_text())
                    if hashlib.sha256(raw).hexdigest() != meta['normalized_payload_sha256']:
                        raise ValueError('Cached raw payload changed')
                    cached_rows = parse_page(payload, start, end)
                    reusable = validate_rows(cached_rows, start, end)['complete']
                except (ValueError, KeyError, TypeError, OSError, EOFError):
                    reusable = False
            if not reusable:
                # Keep every failed/incomplete original, even if the refetch
                # itself fails. A complete cache is the only reusable cache.
                attempt = 1
                while any(folder.joinpath(f'{start}.retry{attempt}'+suffix).exists()
                          for suffix in ('.json.gz', '.meta.json')):
                    attempt += 1
                for path, suffix in ((page_path, '.json.gz'), (meta_path, '.meta.json')):
                    if path.exists():
                        path.replace(folder/f'{start}.retry{attempt}{suffix}')
                payload, meta = get_public('/openApi/swap/v2/quote/klines',
                    {'symbol':pair,'interval':'1m','startTime':start*1000,
                     'endTime':end*1000-1,'limit':1440})
                raw = json.dumps(payload, separators=(',',':'), allow_nan=False).encode()
                meta['normalized_payload_sha256'] = hashlib.sha256(raw).hexdigest()
                page_path.write_bytes(gzip.compress(raw, mtime=0))
                _save_json(meta_path, meta)
                time.sleep(.15)
            rows = parse_page(payload, start, end)
            coverage = validate_rows(rows,start,end)
            result['pages'].append({'start':start,'end':end,**meta,**coverage})
            all_rows.extend(rows)
            if n % 30 == 0:
                print(json.dumps({'symbol':symbol,'pages':n+1,'candles':len(all_rows)}), flush=True)
            if not listed:
                result['errors'].append('Configured symbol not in current contract list; historical probe only')
                break
        except Exception as exc:
            result['errors'].append({'start':start,'end':end,'reason':type(exc).__name__+': '+str(exc)})
            # Persist progress and do not turn an API error into an empty success.
            _save_json(output/'pair_manifests'/(symbol+'.json'), result)
            if not listed:
                break
    result['coverage'] = validate_rows(all_rows, WARM_START, END)
    normalized = json.dumps(all_rows, separators=(',',':'), allow_nan=False).encode()
    result['candles_sha256'] = hashlib.sha256(normalized).hexdigest()
    if all_rows:
        dest = output/(symbol+'.parquet')
        pd.DataFrame(all_rows).to_parquet(dest,index=False)
        result['parquet_sha256'] = hashlib.sha256(dest.read_bytes()).hexdigest()
    # Real funding-rate history includes actual settlement mark price, if served.
    try:
        payload, meta = get_public('/openApi/swap/v2/quote/fundingRate',
              {'symbol':pair,'startTime':WARM_START*1000,'endTime':END*1000-1,'limit':1000})
        _save_json(output/'funding_raw'/(symbol+'.json'), {'payload':payload,'request':meta})
        if payload.get('code') != 0:
            raise ValueError('Funding API code '+str(payload.get('code')))
        funding = []
        for r in payload.get('data') or []:
            if r.get('symbol') != pair:
                raise ValueError('Funding symbol mismatch')
            stamp = int(r['fundingTime'])//1000
            rate, mark = float(r['fundingRate']),float(r['markPrice'])
            if not WARM_START <= stamp < END or not math.isfinite(rate) or not math.isfinite(mark) or mark <= 0:
                raise ValueError('Invalid/outside funding settlement')
            funding.append({'time':stamp,'rate':rate,'mark_price':mark})
        funding.sort(key=lambda r:r['time'])
        if len({x['time'] for x in funding}) != len(funding):
            raise ValueError('Duplicate funding settlement')
        _save_json(output/'funding'/(symbol+'.json'), funding)
        # Unknown schedule: do not claim that an empty/partial response proves zero funding.
        result['funding'] = {'rows':len(funding),'start':funding[0]['time'] if funding else None,
                             'end':funding[-1]['time'] if funding else None,
                             'coverage_certified':False,'request':meta}
    except Exception as exc:
        result['funding'] = {'rows':None,'coverage_certified':False,'error':str(exc)}
    result['status'] = 'complete_candles' if result['coverage']['complete'] and listed else 'incomplete_or_unlisted'
    _save_json(output/'pair_manifests'/(symbol+'.json'), result)
    print(json.dumps({'symbol':symbol,'status':result['status'],'rows':len(all_rows),'errors':len(result['errors'])}),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--universe',required=True, type=Path, help='Frozen JSON [{symbol,listed}]')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=2,choices=(1,2,4))
    args = parser.parse_args()
    universe = json.loads(args.universe.read_text())
    if len({x['symbol'] for x in universe}) != len(universe):
        raise ValueError('Duplicate universe symbol')
    args.output.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda item:collect_pair(item['symbol'],item['listed'],args.output),universe))
    manifest = {'start':START,'end':END,'warm_start':WARM_START,'configured_symbols':len(universe),
                'complete_symbols':sum(x['status']=='complete_candles' for x in results),'pairs':results,
                'spread_evidence':'Historical bid/ask unavailable; do not label assumed spreads observed',
                'universe_selection':'Current configured universe, not point-in-time historical constituents; survivor/selection bias remains'}
    _save_json(args.output/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='pairs'}),flush=True)


if __name__ == '__main__':
    main()
