"""Real public BingX H1/H4 warmup; no account or scanner runtime imports."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
from research_scalping.data import get_public,WARM_START


def validate(rows,start,end,seconds):
    expected=(end-start)//seconds
    times=[r['time'] for r in rows]
    return {'expected_rows':expected,'rows':len(rows),
            'complete':times==list(range(start,end,seconds)),
            'missing_opens':sorted(set(range(start,end,seconds))-set(times))}


def normalize(payload,start,end,seconds):
    if payload.get('code')!=0 or not isinstance(payload.get('data'),list):
        raise ValueError('No successful historical candle list')
    rows=[]
    for item in payload['data']:
        stamp=int(item['time'])
        if stamp%1000 or (stamp//1000)%seconds or not start*1000<=stamp<end*1000:
            raise ValueError('HTF candle out of range or grid')
        r={'time':stamp//1000,**{k:float(item[k]) for k in ('open','high','low','close','volume')}}
        o,h,l,c,v=(r[k] for k in ('open','high','low','close','volume'))
        if not all(math.isfinite(x) for x in (o,h,l,c,v)) or min(o,h,l,c)<=0 or v<0 or not l<=min(o,c)<=max(o,c)<=h:
            raise ValueError('Malformed HTF candle')
        rows.append(r)
    rows.sort(key=lambda r:r['time'])
    if len({r['time'] for r in rows})!=len(rows):
        raise ValueError('Duplicated HTF candle')
    return rows


def collect_one(symbol,interval,seconds,out):
    import pandas as pd
    start,end=WARM_START-35*86400,WARM_START
    raw_path=out/'raw'/(symbol+'_'+interval+'.json.gz')
    meta_path=out/'raw'/(symbol+'_'+interval+'.meta.json')
    raw_path.parent.mkdir(parents=True,exist_ok=True)
    prior=[]
    for attempt in range(3):
        if attempt==0 and raw_path.exists() and meta_path.exists():
            raw=gzip.decompress(raw_path.read_bytes()); payload=json.loads(raw); meta=json.loads(meta_path.read_text())
            if hashlib.sha256(raw).hexdigest()!=meta['normalized_response_sha256']:
                raise ValueError('Cached HTF raw hash mismatch')
        else:
            payload,meta=get_public('/openApi/swap/v2/quote/klines',{'symbol':symbol[:-4]+'-USDT','interval':interval,
                                    'startTime':start*1000,'endTime':end*1000-1,'limit':1000})
            raw=json.dumps(payload,separators=(',',':'),allow_nan=False).encode()
            if raw_path.exists():
                backup=raw_path.with_name(raw_path.name+'.attempt'+str(attempt))
                backup.write_bytes(raw_path.read_bytes())
            raw_path.write_bytes(gzip.compress(raw,mtime=0))
            meta['normalized_response_sha256']=hashlib.sha256(raw).hexdigest()
            meta_path.write_text(json.dumps(meta,indent=2))
        try:
            rows=normalize(payload,start,end,seconds)
            coverage=validate(rows,start,end,seconds)
            if coverage['complete']:
                path=out/(symbol+'_'+interval+'.parquet')
                pd.DataFrame(rows).to_parquet(path,index=False)
                return {'symbol':symbol,'interval':interval,'seconds':seconds,'status':'complete',
                        'coverage':coverage,'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'request':meta,'previous_failures':prior}
            prior.append({'attempt':attempt,'coverage':coverage})
        except (ValueError,KeyError,TypeError) as ex:
            prior.append({'attempt':attempt,'error':str(ex)})
        time.sleep(1)
    return {'symbol':symbol,'interval':interval,'status':'NOT_EVALUABLE','failures':prior,'request':meta}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.data/'manifest.json').read_text())
    tasks=[(r['symbol'],tf,seconds,args.output) for r in manifest['pairs'] if r['status']=='complete_candles'
           for tf,seconds in [('1h',3600),('4h',14400)]]
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(lambda x:collect_one(*x),tasks):
            results.append(result)
            (args.output/'progress.json').write_text(json.dumps(results,indent=2))
            print(json.dumps({k:v for k,v in result.items() if k not in ('request','previous_failures')}),flush=True)
    final={'source':'BingX native H1/H4 public history, warmup only','start':WARM_START-35*86400,'end':WARM_START,
           'requested':len(tasks),'complete':sum(r['status']=='complete' for r in results),'pairs':results}
    (args.output/'manifest.json').write_text(json.dumps(final,indent=2))
    print('SUMMARY',json.dumps({k:v for k,v in final.items() if k!='pairs'}),flush=True)


if __name__=='__main__': main()
