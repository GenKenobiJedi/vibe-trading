"""Synthetic midnight accounting; distinguish candle close from next open."""
import numpy as np
import pandas as pd
import pytest
from research_scalping.execution import CostConfig, run
from research_ict.reporting import groups


def midnight(side, mode, *, phases=True):
    start=int(pd.Timestamp('2026-06-30T23:56:00Z').timestamp())
    boundary=int(pd.Timestamp('2026-07-01T00:00:00Z').timestamp())
    df=pd.DataFrame(dict(time=np.arange(start,start+600,60,dtype=np.int64),
                         open=100.,high=100.,low=100.,close=100.,volume=1.,atr=1.))
    if mode=='open_stop':
        df.loc[4:,['open','high','low','close']]=100-side
    elif mode=='close_stop':
        df.loc[3,'low' if side==1 else 'high']=100-side
    signal=np.zeros(len(df),dtype=np.int8)
    signal[0]=side
    config=CostConfig(fee=0.,slippage=0.,spread_bps=0.,stop_atr=1.,target_atr=2.,
                      max_hold_bars=30,cooldown_bars=0)
    kwargs={'record_equity_phases':True} if phases else {}
    result=run(df,signal,config,start=start,end=start+600,
               funding=[dict(time=boundary,rate=.01,mark_price=100.)] if mode=='funding' else None,
               **kwargs)
    return result,dict(window_start=start,window_end=start+600),boundary


def monthly(result,metadata):
    return {row['month']:row for row in groups(result,metadata) if row['direction']=='BOTH'}


@pytest.mark.parametrize('side',[1,-1])
@pytest.mark.parametrize('mode',['funding','open_stop','close_stop'])
def test_midnight_funding_open_and_completed_candle_are_different_months(side,mode):
    result,metadata,boundary=midnight(side,mode)
    assert len(result['equity_phases'])==len(result['equity'])
    months=monthly(result,metadata)
    june,july=months['2026-06'],months['2026-07']
    if mode=='close_stop':
        assert (june['mtm_net_pnl'],july['mtm_net_pnl'])==pytest.approx((-10.,0.))
        assert (june['max_drawdown_usdt'],july['max_drawdown_usdt'])==pytest.approx((10.,0.))
        assert (june['N'],july['N'])==(1,0)
    else:
        loss=10.*side if mode=='funding' else 10.
        assert (june['mtm_net_pnl'],july['mtm_net_pnl'])==pytest.approx((0.,-loss))
        assert june['max_drawdown_usdt']==pytest.approx(0.)
        assert july['max_drawdown_usdt']==pytest.approx(max(loss,0.))
        assert (june['N'],july['N'])==(0,1)
    assert june['mtm_net_pnl']+july['mtm_net_pnl']==pytest.approx(result['equity'][-1][1]-1000.)


@pytest.mark.parametrize('side',[1,-1])
@pytest.mark.parametrize('mode',['funding','open_stop','close_stop'])
def test_event_phase_annotation_does_not_change_any_fill_or_equity(side,mode):
    marked,_,_=midnight(side,mode)
    plain,_,_=midnight(side,mode,phases=False)
    assert marked['equity']==plain['equity']
    assert marked['rejections']==plain['rejections']
    assert [{k:v for k,v in trade.items() if k!='exit_event_phase'}
            for trade in marked['trades']]==plain['trades']


def test_legacy_duplicate_boundary_without_phase_evidence_is_not_silently_allocated():
    result,metadata,_=midnight(1,'funding',phases=False)
    with pytest.raises(ValueError,match='phase'):
        groups(result,metadata)


@pytest.mark.parametrize('bad_phase', [None, 'unknown'])
def test_equal_boundary_equity_does_not_disambiguate_legacy_exit_month(bad_phase):
    boundary=int(pd.Timestamp('2026-07-01T00:00:00Z').timestamp())
    trade=dict(net_pnl=5.,exit_time=boundary,entry_time=boundary-60,side='LONG',
               net_r=.5,holding_seconds=60,exit_reason='terminal',fees=0.,
               slippage_cost=0.,spread_cost=0.,funding_cost=0.)
    if bad_phase is not None:
        trade['exit_event_phase']=bad_phase
    result=dict(trades=[trade],equity=[[boundary-120,1000.],[boundary,1005.],
                                    [boundary,1005.],[boundary+120,1005.]])
    with pytest.raises(ValueError,match='phase'):
        groups(result,dict(window_start=boundary-120,window_end=boundary+120))


def test_phase_evidence_cannot_be_shorter_than_equity():
    result,metadata,_=midnight(1,'funding')
    result['equity_phases'].pop()
    with pytest.raises(ValueError,match='phase'):
        groups(result,metadata)
