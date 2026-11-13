"""Parent integration regressions; synthetic gates, not strategy PnL."""
import numpy as np
import pandas as pd
import pytest
from research_ict import features as f


def sequence(monkeypatch,sign=1,outside=False,chase=False):
    n=12
    df=pd.DataFrame(dict(time=np.arange(n)*60,open=106.,high=107.,low=105.,close=106.,volume=10.))
    df.loc[2,['open','high','low','close']]=[101.,102.,100.,101.5]
    df.loc[3,['open','high','low','close']]=[106.,110.,105.,109.]
    df.loc[4,['open','high','low','close']]=[108.,109.,107.,108.]
    df.loc[5,['open','high','low','close']]=[103.,103.1,102.3,102.5]
    bottom,top=(101.7,101.9) if outside else (102.1,102.2)
    df.loc[6,['open','high','low','close']]=[102.2,106. if chase else 104.,101.8 if outside else 102.15,105. if chase else 103.5]
    hp,lp=np.full(n,np.nan),np.full(n,np.nan)
    hp[0],lp[0]=108.,101.
    zone=(1,bottom,top,4,3,'ob')
    if sign<0:
        o,h,low,c=(df[k].to_numpy(copy=True) for k in ('open','high','low','close'))
        df['open'],df['high'],df['low'],df['close']=220-o,220-low,220-h,220-c
        hp,lp=220-lp,220-hp
        zone=(-1,220-top,220-bottom,4,3,'ob')
    monkeypatch.setattr(f,'_pivots',lambda data,window:(hp,lp))
    monkeypatch.setattr(f,'_formations',lambda data:{4:[zone]})
    return df


@pytest.mark.parametrize('sign',[1,-1])
def test_ote_seen_does_not_authorize_unrelated_zone_outside_ote(monkeypatch,sign):
    df=sequence(monkeypatch,sign,outside=True)
    result=f._sequence(df,np.full(len(df),sign))
    assert not result['fresh_zone_'+('long' if sign>0 else 'short')].any()


@pytest.mark.parametrize('sign',[1,-1])
def test_reaction_cannot_chase_far_beyond_frozen_ote(monkeypatch,sign):
    df=sequence(monkeypatch,sign,chase=True)
    result=f._sequence(df,np.full(len(df),sign))
    assert not result['fresh_zone_'+('long' if sign>0 else 'short')].any()


@pytest.mark.parametrize('sign',[1,-1])
def test_fresh_first_tap_inside_ote_still_allowed(monkeypatch,sign):
    df=sequence(monkeypatch,sign)
    result=f._sequence(df,np.full(len(df),sign))
    assert result['fresh_zone_'+('long' if sign>0 else 'short')][6]


@pytest.mark.parametrize('origin,eligible',[(10,False),(195,True)])
def test_htf_diagnostic_zone_preview_not_automatically_fresh(origin,eligible):
    n=200
    df=pd.DataFrame(dict(time=np.arange(n)*3600,open=100.,high=101.,low=99.,close=100.,volume=10.))
    zone={'type':'bullish','active':True,'confirmed':True,'index':origin,'formed_at':origin,
          'known_at':origin,'confirmed_at':origin,'gap_bottom':99.,'gap_top':99.5}
    def scanner(*prices,**kwargs):
        return {'score':8,'details':{'fvgs':[zone],'obs':[]}}
    _,contexts,_=f._scan_context(df,scanner,3600)
    assert bool(contexts[-1][0]) is eligible


@pytest.mark.parametrize('sign', [1, -1])
@pytest.mark.parametrize('valid_first', [True, False])
def test_later_nonoverlapping_zone_cannot_erase_valid_first_tap(monkeypatch, sign, valid_first):
    df=sequence(monkeypatch, sign)
    df.loc[6,'low' if sign == 1 else 'high']=102.05 if sign == 1 else 117.95
    zones=[(1,101.9,102.2,4,3,'ob'),(1,101.8,102.07,4,3,'fvg')]
    if sign == -1:
        zones=[(-1,220-top,220-bottom,known,origin,kind)
               for _,bottom,top,known,origin,kind in zones]
    if not valid_first:
        zones.reverse()
    monkeypatch.setattr(f,'_formations',lambda data:{4:zones})
    result=f._sequence(df,np.full(len(df),sign,dtype=np.int8))
    assert result['fresh_zone_'+('long' if sign == 1 else 'short')][6]
