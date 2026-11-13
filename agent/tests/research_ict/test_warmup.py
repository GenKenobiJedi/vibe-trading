from research_ict.warmup import normalize,validate
import pytest


def row(t):
    return {'time':t*1000,'open':'2','high':'3','low':'1','close':'2.5','volume':'4'}


def test_complete_real_htf_and_no_forward_fill():
    a=normalize({'code':0,'data':[row(3600),row(0)]},0,7200,3600)
    assert [r['time'] for r in a]==[0,3600]
    assert validate(a,0,7200,3600)['complete']
    assert not validate(a[:1],0,7200,3600)['complete']


def test_wrong_range_grid_and_duplicate_rejected():
    for raw in [[row(60)],[row(7200)],[row(0),row(0)]]:
        with pytest.raises(ValueError): normalize({'code':0,'data':raw},0,7200,3600)


def test_ohlc_and_api_errors_not_fabricated():
    with pytest.raises(ValueError): normalize({'code':109418,'data':[]},0,7200,3600)
    r=row(0); r['high']='0'
    with pytest.raises(ValueError): normalize({'code':0,'data':[r]},0,7200,3600)
