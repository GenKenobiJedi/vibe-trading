from research_ict.funding import normalize
import pytest


def test_signed_rate_and_observed_mark_not_zero_fabrication():
    payload={'code':0,'data':[{'symbol':'SOL-USDT','fundingTime':3600000,'fundingRate':'-0.0002','markPrice':'99.5'}]}
    rows=normalize(payload,'SOL-USDT',0,7200)
    assert rows==[{'time':3600,'rate':-.0002,'mark_price':99.5}]
    assert normalize({'code':0,'data':None},'SOL-USDT',0,7200)==[]


def test_bad_mark_symbol_api_duplicates_refused():
    base={'symbol':'SOL-USDT','fundingTime':3600000,'fundingRate':'0.0001','markPrice':'99.5'}
    for bad in [dict(base,markPrice='0'),dict(base,symbol='BTC-USDT'),dict(base,fundingTime=7200000)]:
        with pytest.raises(ValueError): normalize({'code':0,'data':[bad]},'SOL-USDT',0,7200)
    with pytest.raises(ValueError): normalize({'code':0,'data':[base,base]},'SOL-USDT',0,7200)
    with pytest.raises(ValueError): normalize({'code':109418,'data':[]},'SOL-USDT',0,7200)
