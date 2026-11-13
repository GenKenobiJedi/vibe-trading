"""Monthly public funding evidence; original minute data remain untouched."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import shutil
from research_scalping.data import get_public,START,END


def normalize(payload,symbol,start,end):
    if payload.get('code')!=0:
        raise ValueError('Funding API code '+str(payload.get('code')))
    raw=payload.get('data') or []
    if not isinstance(raw,list): raise ValueError('Funding list required')
    rows=[]
    for r in raw:
        stamp=int(r['fundingTime'])
        rate,mark=float(r['fundingRate']),float(r['markPrice'])
        if (r.get('symbol')!=symbol or stamp%1000 or not start*1000<=stamp<end*1000
                or not math.isfinite(rate) or not math.isfinite(mark) or mark<=0):
            raise ValueError('Invalid funding record')
        rows.append({'time':stamp//1000,'rate':rate,'mark_price':mark})
    rows.sort(key=lambda r:r['time'])
    if len({r['time'] for r in rows})!=len(rows): raise ValueError('Duplicate funding event')
    return rows


def windows(start,end):
    t=start
    while t<end:
        d=dt.datetime.fromtimestamp(t,dt.timezone.utc)
        nxt=dt.datetime(d.year+1,1,1,tzinfo=dt.timezone.utc) if d.month==12 else dt.datetime(d.year,d.month+1,1,tzinfo=dt.timezone.utc)
        stop=min(end,int(nxt.timestamp()))
        yield t,stop
        t=stop


def collect_one(pair,out):
    symbol=pair['symbol']; venue=symbol[:-4]+'-USDT'
    result=dict(pair)
    data=[]; pages=[]
    for start,end in windows(START,END):
        path=out/'funding_raw'/symbol/(str(start)+'.json')
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            saved=json.loads(path.read_text()); payload=saved['payload']; meta=saved['request']
            expected=saved['normalized_payload_sha256']
            if hashlib.sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()!=expected:
                raise ValueError('Cached funding hash changed')
        else:
            payload,meta=get_public('/openApi/swap/v2/quote/fundingRate',{'symbol':venue,
                'startTime':start*1000,'endTime':end*1000-1,'limit':1000})
            body=json.dumps(payload,separators=(',',':')).encode()
            path.write_text(json.dumps({'payload':payload,'request':meta,'normalized_payload_sha256':hashlib.sha256(body).hexdigest()},indent=2))
        rows=normalize(payload,venue,start,end)
        if len(rows)>=1000:
            raise ValueError('Funding monthly response reached cap; split before claiming full retrieval')
        pages.append({'start':start,'end':end,'rows':len(rows),'file':str(path.relative_to(out)),
                      'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'request':meta})
        data.extend(rows)
    if len({r['time'] for r in data})!=len(data): raise ValueError('Funding windows overlap')
    dest=out/'funding'/(symbol+'.json'); dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(data,indent=2))
    gaps=[b['time']-a['time'] for a,b in zip(data,data[1:])]
    result['funding']={'rows':len(data),'start':data[0]['time'] if data else None,'end':data[-1]['time'] if data else None,
         'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'coverage_certified':False,
         'bounded_api_windows_retrieved':True,'schedule_basis':'Historical per-contract settlement schedule not independently available; actual API events only',
         'max_observed_gap_seconds':max(gaps,default=None),'pages':pages}
    print(json.dumps({'symbol':symbol,'funding_rows':len(data),'monthly_pages':len(pages),'max_observed_gap_seconds':max(gaps,default=None)}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3); args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    original=json.loads((args.source/'manifest.json').read_text())
    active=[r for r in original['pairs'] if r['status']=='complete_candles']
    for pair in active:
        source=args.source/(pair['symbol']+'.parquet'); target=args.output/source.name
        expected=pair['parquet_sha256']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=expected: raise ValueError('Original minute data changed')
        if not target.exists(): shutil.copy2(source,target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=expected: raise ValueError('Copied minute data hash mismatch')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results=list(pool.map(lambda r:collect_one(r,args.output),active))
    by={r['symbol']:r for r in results}
    enriched={**original,'pairs':[by.get(r['symbol'],r) for r in original['pairs']],
              'original_manifest_sha256':hashlib.sha256((args.source/'manifest.json').read_bytes()).hexdigest(),
              'original_candle_evidence_root':str(args.source),'funding_retrieval':'Bounded calendar-month GETs below response cap; no fake/zero-filled settlements'}
    (args.output/'manifest.json').write_text(json.dumps(enriched,indent=2))
    print('SUMMARY',json.dumps({'coins':len(results),'funding_events':sum(r['funding']['rows'] for r in results),'unchanged_minute_parquets':len(active)}),flush=True)


if __name__=='__main__': main()
