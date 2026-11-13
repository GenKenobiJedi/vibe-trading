"""Runner lane fixtures inject features; they never claim financial evidence."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_ict import runner as r
from research_ict.reporting import digest, load_json, verify_seal


@pytest.fixture
def lab(tmp_path):
    data, output, htf = (tmp_path/x for x in ('data','out','htf'))
    data.mkdir(); htf.mkdir()
    start = 1781308800  # Synthetic dates only, aligned to 4h.
    edges = [start+i*40*60 for i in range(6)]
    bounds = dict(train=[edges[0],edges[3]], wf1_fit=[edges[0],edges[1]],
                  wf1_test=[edges[1],edges[2]], wf2_fit=[edges[0],edges[2]],
                  wf2_test=[edges[2],edges[3]], validation=[edges[3],edges[4]],
                  holdout=[edges[4],edges[5]], full5months=[edges[0],edges[5]])
    times = np.arange(start-600, edges[5], 60, dtype=np.int64)
    df = pd.DataFrame(dict(time=times, open=100., high=100.1, low=99.9, close=100., volume=10.))
    df.to_parquet(data/'SYNTH.parquet', index=False)
    pairs = [dict(symbol='SYNTH', status='complete_candles', parquet_sha256=r.file_hash(data/'SYNTH.parquet'),
                  funding={'coverage_certified':False}), dict(symbol='MATICUSDT', status='incomplete_or_unlisted')]
    (data/'manifest.json').write_text(json.dumps(dict(pairs=pairs, warm_start=start-600, synthetic=True)))
    (data/'funding').mkdir()
    (data/'funding'/'SYNTH.json').write_text('[]')
    for label, step in [('1h',3600),('4h',14400)]:
        end = (start-600)//step*step
        prelude = pd.DataFrame(dict(time=np.arange(end-220*step,end,step,dtype=np.int64),
                                   open=100., high=101., low=99., close=100., volume=10.))
        prelude.to_parquet(htf/f'SYNTH_{label}.parquet',index=False)
    neuro = tmp_path/'ict.py'
    neuro.write_text('def ict_scan(*args, **kwargs):\n    return {}\n')
    source = tmp_path/'fake_features.py'
    source.write_text('''import numpy as np
calls = []
seal_path = None
holdout_start = None
def build_features(df, *, ict_module_path, htf_warmup=None):
    from pathlib import Path
    assert Path(ict_module_path).is_file()
    if holdout_start is not None and df.time.max() >= holdout_start:
        assert Path(seal_path).is_file(), 'holdout features before validation seal'
    calls.append(int(df.time.max()))
    f = df.copy()
    for name in ('atr','atr1m'): f[name] = 1.
    for name in ('ema9','ema21'): f[name] = 100.
    for name in ('momentum','trend5','trend1h','trend4h'): f[name] = np.int8(0)
    f['adx5'] = 10.
    for name in ('ict_score1h_long','ict_score1h_short','ict_score4h_long','ict_score4h_short'): f[name] = 0.
    for side in ('long','short'):
        for name in ('baseline','sweep','ote','fresh_zone','reaction','pullback','deadfish','strong_trend'):
            f[name+'_'+side] = False
    f.attrs.update(provenance={'synthetic':True}, counts={'synthetic_rows':len(f)})
    return f
def signals(frame, variant, params=None):
    assert params == {'volume_ratio':1.2}
    out = np.zeros(len(frame),dtype=np.int8)
    out[::7] = 1
    return out
''')
    spec = importlib.util.spec_from_file_location('synthetic_features', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.seal_path = str(output/'validation_decision.json')
    module.holdout_start = bounds['holdout'][0]
    return dict(data_dir=data, output=output, neuro_root=neuro, htf_dir=htf,
                feature_module=module, bounds=bounds, workers=2)


def test_catalog_fixed_shared_risk_and_components():
    arms = r.catalog()
    assert {a['name'] for a in arms} == {'A_ict_corrected','B_sweep_ote','C_deadfish',
        'C_momentum','C_time','D_ha_bb_rsi','E_B_deadfish','E_B_pullback','E_B_pullback_deadfish'}
    for arm in arms:
        c = r.cost_config(arm)
        assert (c.stop_atr,c.target_atr,c.trailing_atr,c.cooldown_bars,c.max_hold_bars,
                c.risk_usdt,c.leverage,c.min_net_rr) == (2.,3.,2.,5,30,10.,10.,1.)
        assert arm['params'] == {'volume_ratio':1.2}


def test_preview_and_stress_preserve_opportunities():
    frame = pd.DataFrame(dict(time=np.arange(4)*60, open=100.,high=100.,low=100.,close=100.,atr=[1.,.001,1.,1.]))
    raw = np.array([1,1,-1,0],dtype=np.int8)
    signal, stops, targets, counts = r.plan(frame, raw)
    assert signal.tolist() == [1,0,-1,0]
    assert raw.tolist() == [1,1,-1,0]
    assert counts['preview_min_net_rr'] == 1
    assert stops[0] == 98. and targets[0] == 103.


def test_synthetic_train_eval_report_and_sealed_stage_order(lab):
    r.execute_phase(phase='train', **lab)
    assert max(lab['feature_module'].calls) < lab['bounds']['train'][1]
    selection = verify_seal(lab['output']/'selection.json')
    assert selection['wf1_winner'] == selection['wf2_winner'] == 'A_ict_corrected'
    r.execute_phase(phase='eval', **lab)
    decision = verify_seal(lab['output']/'validation_decision.json')
    assert decision['winner']=='A_ict_corrected'
    assert decision['adopted_components']==[]
    assert max(lab['feature_module'].calls) == lab['bounds']['holdout'][1]-60
    report = r.execute_phase(phase='report', **lab)
    assert report['live_adoption'] is False
    assert report['selected'] == 'A_ict_corrected'
    assert [x['phase'] for x in report['invocations']]==['train','eval','report']
    assert report['evidence_class']=='synthetic engineering fixture; not financial evidence'
    rows = load_json(lab['output']/'holdout_groups.json')
    assert any(x['coin']=='MATICUSDT' and x['strategy']=='E_selected' and x['N'] is None for x in rows)
    assert {x['strategy'] for x in rows} == {a['name'] for a in r.catalog()}|{'E_selected'}
    assert (lab['output']/'holdout_groups.csv').exists()
    ledger = next((lab['output']/'runs'/'holdout').rglob('*.json.gz'))
    result = load_json(ledger)
    assert result['equity'][-1][1] == pytest.approx(1000+sum(t['net_pnl'] for t in result['trades']))
    assert result['funding_evidence']['state']=='present_uncertified'
    assert result['opportunities_sha256']
    calls = len(lab['feature_module'].calls)
    r.execute_phase(phase='eval', **lab)  # Verified checkpoints resume, no rewrite.
    assert len(lab['feature_module'].calls)==calls


@pytest.mark.parametrize('target', ['funding','missing_funding','parquet','htf','neuro','feature','parameters'])
def test_inputs_tamper_refused_before_eval_features(lab, target, monkeypatch):
    r.execute_phase(phase='train', **lab)
    calls = len(lab['feature_module'].calls)
    if target=='funding': (lab['data_dir']/'funding'/'SYNTH.json').write_text('[{}]')
    if target=='missing_funding': (lab['data_dir']/'funding'/'MATICUSDT.json').write_text('[]')
    if target=='parquet':
        with (lab['data_dir']/'SYNTH.parquet').open('ab') as f: f.write(b'tamper')
    if target=='htf':
        with (lab['htf_dir']/'SYNTH_1h.parquet').open('ab') as f: f.write(b'tamper')
    if target=='neuro': lab['neuro_root'].write_text('# changed\n')
    if target=='feature': Path(lab['feature_module'].__file__).write_text('# changed\n')
    if target=='parameters':
        old = r.catalog
        def changed():
            arms = old(); arms[0]['params']['volume_ratio']=1.3; return arms
        monkeypatch.setattr(r,'catalog',changed)
    with pytest.raises(ValueError, match='changed|fingerprint'):
        r.execute_phase(phase='eval', **lab)
    assert len(lab['feature_module'].calls)==calls


def test_selection_requires_each_component_and_positive_baseline_delta():
    def row(name, net=20., n=120, dd=5.):
        return dict(name=name, N=n, net_pnl=net, profit_factor=1.2,max_drawdown_usdt=dd)
    rows = [row(a['name']) for a in r.catalog()]
    by = {x['name']:x for x in rows}
    by['A_ict_corrected']['net_pnl']=10.
    by['E_B_pullback']['net_pnl']=30.
    by['D_ha_bb_rsi']['net_pnl']=-1.
    assert r.choose(rows)['winner']=='E_B_deadfish'
    by['C_deadfish']['N']=0
    assert r.choose(rows)['winner']=='A_ict_corrected'


def test_workers_bounded_and_explicit_module_path(lab):
    with pytest.raises(ValueError, match='workers'):
        r.execute_phase(phase='train', **{**lab,'workers':3})
    with pytest.raises(ValueError, match='module'):
        r.execute_phase(phase='train', **{**lab,'neuro_root':lab['neuro_root'].parent})


def test_fit_prefixes_and_test_selection_snapshots(lab):
    r.execute_phase(phase='train', **lab)
    assert lab['feature_module'].calls == [lab['bounds'][name][1]-60 for name in ('wf1_fit','wf2_fit','train')]
    for name in ('wf1_fit','wf2_fit','train'):
        fit = verify_seal(lab['output']/f'{name}_selection.json')
        complete = verify_seal(lab['output']/f'{name}_complete.json')
        assert fit['fit_window']==name
        assert complete['bounds']==lab['bounds'][name]
    r.execute_phase(phase='eval', **lab)
    for name in ('wf1_test','wf2_test'):
        stage = verify_seal(lab['output']/f'{name}_complete.json')
        assert stage['selected']=='A_ict_corrected'


def test_ledger_and_checkpoint_tamper_refused_on_resume(lab):
    r.execute_phase(phase='train', **lab)
    path = next((lab['output']/'runs'/'train').rglob('*.json.gz'))
    with path.open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        r.execute_phase(phase='eval', **lab)


def test_selection_components_need_train_validation_and_walkforward_help():
    def evidence():
        return [dict(name=arm['name'],N=200,net_pnl=10. if arm['name']=='A_ict_corrected' else 30.,
                     profit_factor=1.3,max_drawdown_usdt=10.) for arm in r.catalog()]
    training, validation, wf = evidence(), evidence(), evidence()
    assert r.choose(training,validation,wf)['winner']!='A_ict_corrected'
    for row in wf:
        if row['name'] in ('C_deadfish','D_ha_bb_rsi'): row['net_pnl']=10.
    assert r.choose(training,validation,wf)['winner']=='A_ict_corrected'
    for row in wf: row['net_pnl']=30.
    for row in training: row['N']=0; row['profit_factor']=None
    assert r.choose(training,validation,wf)['winner']=='A_ict_corrected'


@pytest.mark.parametrize('scenario',['fee2','slip5','combined5bps'])
def test_stress_reuses_exact_base_opportunities_and_cost_reconciles(lab,scenario):
    r.execute_phase(phase='train', **lab)
    r.execute_phase(phase='eval', **lab)
    folder = lab['output']/'runs'/'holdout'/'SYNTH'/'A_ict_corrected'
    base, stress = load_json(folder/'base.json.gz'), load_json(folder/f'{scenario}.json.gz')
    assert base['opportunities_sha256']==stress['opportunities_sha256']
    for result in (base,stress):
        for trade in result['trades']:
            assert trade['net_pnl']==pytest.approx(trade['gross_pnl']-trade['fees']-trade['funding_cost'])
            assert trade['initial_risk_usdt']<=10.+1e-9
        assert result['equity'][-1][1]==pytest.approx(1000.+sum(t['net_pnl'] for t in result['trades']))
    assert stress['equity'][-1][1] <= base['equity'][-1][1]


def test_wfoos_carries_cash_and_uses_preceding_fit_choices(lab):
    r.execute_phase(phase='train', **lab)
    r.execute_phase(phase='eval', **lab)
    result = load_json(lab['output']/'runs'/'wfoos'/'SYNTH'/'E_selected'/'base.json.gz')
    first, second = result['folds']
    assert first['selection_fit']=='wf1_fit' and second['selection_fit']=='wf2_fit'
    assert first['initial_equity']==1000.
    assert second['initial_equity']==pytest.approx(first['final_equity'])
    assert second['initial_equity']<1000.
    assert result['equity'][-1][1]==pytest.approx(1000.+sum(t['net_pnl'] for t in result['trades']))
    assert result['metadata']['policy']=='walkforward_carry_cash_force_flat_fold_boundary'
    assert result['feature_counts']['deadfish_long']==0
    assert result['funding_evidence']['supplied_settlements']==0
    assert result['funding_evidence']['coverage_certified'] is False


def test_stale_imported_feature_code_cannot_claim_changed_source(lab):
    source = Path(lab['feature_module'].__file__)
    source.write_text(source.read_text().replace('out[::7] = 1','out[::8] = 1'))
    with pytest.raises(ValueError, match='[Ll]oaded.*source|source.*loaded'):
        r.execute_phase(phase='train', **lab)


def test_validation_seal_survives_interrupted_holdout_and_resume(lab, monkeypatch):
    r.execute_phase(phase='train', **lab)
    original = r._load_frame
    def interrupt(registration,symbol,end,features,neuro):
        if end==lab['bounds']['holdout'][1]:
            assert (lab['output']/'validation_decision.json').exists()
            raise RuntimeError('synthetic interruption after validation seal')
        return original(registration,symbol,end,features,neuro)
    monkeypatch.setattr(r,'_load_frame',interrupt)
    with pytest.raises(RuntimeError,match='synthetic interruption'):
        r.execute_phase(phase='eval', **lab)
    checksum = r.file_hash(lab['output']/'validation_decision.json')
    assert not (lab['output']/'holdout_complete.json').exists()
    calls = len(lab['feature_module'].calls)
    monkeypatch.setattr(r,'_load_frame',original)
    r.execute_phase(phase='eval', **lab)
    assert r.file_hash(lab['output']/'validation_decision.json')==checksum
    assert len(lab['feature_module'].calls)==calls+2  # Holdout + continuous diagnostic only.


def test_b_structural_stop_can_only_widen_and_is_cost_gated():
    frame = pd.DataFrame(dict(time=np.arange(3)*60,open=100.,high=100.,low=100.,close=100.,atr=1.,
                              structural_stop_long=[99.,97.5,90.]))
    signal,stops,targets,counts = r.plan(frame,np.ones(3,dtype=np.int8),structural=True)
    assert stops.tolist()==[98.,97.5,90.]
    assert signal.tolist()==[1,1,0]
    assert targets.tolist()==[103.,103.,103.]
    assert counts['preview_min_net_rr']==1


def test_split_chronology_refused(lab):
    bounds = dict(lab['bounds']); bounds['validation']=bounds['holdout']
    with pytest.raises(ValueError,match='chronology'):
        r.execute_phase(phase='train', **{**lab,'bounds':bounds})


def test_fingerprints_stable_across_fresh_python_hash_seeds(lab):
    import subprocess
    import sys
    script = '''import importlib.util, sys
from pathlib import Path
from research_ict.runner import _code
from research_ict.reporting import digest
spec = importlib.util.spec_from_file_location('synthetic_features',sys.argv[1])
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)
print(digest(_code(f,Path(sys.argv[2]))))
'''
    expected = digest(r._code(lab['feature_module'],lab['neuro_root']))
    for seed in ('1','99'):
        result = subprocess.run([sys.executable,'-c',script,lab['feature_module'].__file__,str(lab['neuro_root'])],
            check=True,capture_output=True,text=True,
            env={'PYTHONPATH':str(Path(r.__file__).resolve().parents[1]),'PYTHONHASHSEED':seed})
        assert result.stdout.strip()==expected


def test_bankrupt_wfoos_sleeve_does_not_reset_cash_or_trade_next_fold(lab):
    event = dict(time=lab['bounds']['wf1_test'][0]+600,rate=10.,mark_price=100.)
    (lab['data_dir']/'funding'/'SYNTH.json').write_text(json.dumps([event]))
    r.execute_phase(phase='train', **lab)
    r.execute_phase(phase='eval', **lab)
    result = load_json(lab['output']/'runs'/'wfoos'/'SYNTH'/'E_selected'/'base.json.gz')
    assert result['folds'][1]['initial_equity']<0
    assert result['folds'][1]['final_equity']==result['folds'][1]['initial_equity']
    assert result['rejections']['bankrupt_sleeve']==1
    assert all(t['entry_time']<lab['bounds']['wf2_test'][0] for t in result['trades'])
    assert result['equity'][-1][1]==pytest.approx(1000.+sum(t['net_pnl'] for t in result['trades']))


def test_training_candidate_not_labeled_validated_adoption():
    rows = [dict(name=a['name'],N=200,net_pnl=10. if a['name']=='A_ict_corrected' else 20.,
                 profit_factor=1.2,max_drawdown_usdt=5.) for a in r.catalog()]
    provisional = r.choose(rows)
    assert provisional['winner']!='A_ict_corrected'
    assert provisional['adopted_components']==[]
    assert provisional['candidate_components']
    assert 'provisional' in provisional['adoption']


def test_neuro_module_symlink_has_one_canonical_identity(lab):
    link = lab['neuro_root'].with_name('declared_ict.py')
    link.symlink_to(lab['neuro_root'])
    r.execute_phase(phase='train', **{**lab,'neuro_root':link})
    registration = verify_seal(lab['output']/'registration.json')
    assert str(lab['neuro_root'].resolve()) in registration['code']['sources']
