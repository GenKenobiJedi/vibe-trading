"""Frozen, offline ICT research train/eval/report driver; no live adoption.

Only a declared feature module builds signals. No production scanner, model,
account, exchange client or previous study is imported by this runner.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import datetime as dt
import hashlib
import importlib
import importlib.metadata
from pathlib import Path
import re
import sys
from types import CodeType

import numpy as np
import pandas as pd

from research_scalping.execution import CostConfig, net_reward_risk, run
from . import reporting as report
from .reporting import atomic_csv, atomic_json, digest, file_hash, load_json, seal, verify_seal

BASELINE = 'A_ict_corrected'
COMPONENTS = {
    'E_B_deadfish': ['B_sweep_ote', 'C_deadfish'],
    'E_B_pullback': ['B_sweep_ote', 'D_ha_bb_rsi'],
    'E_B_pullback_deadfish': ['B_sweep_ote', 'D_ha_bb_rsi', 'C_deadfish'],
}
REQUIRED = {'atr', 'atr1m', 'ema9', 'ema21', 'momentum', 'adx5', 'trend5', 'trend1h', 'trend4h',
            *(f'ict_score{tf}_{side}' for tf in ('1h', '4h') for side in ('long', 'short')),
            *(f'{flag}_{side}' for flag in ('baseline', 'sweep', 'ote', 'fresh_zone', 'reaction',
                                           'pullback', 'deadfish', 'strong_trend') for side in ('long', 'short'))}
SCENARIOS = {'base': {}, 'fee2': {'fee': .001}, 'slip5': {'slippage': .0007},
             'combined5bps': {'fee': .001, 'slippage': .0007, 'spread_bps': 5.}}


def timestamp(date: str) -> int:
    return int(dt.datetime.fromisoformat(date).replace(tzinfo=dt.timezone.utc).timestamp())


def period_bounds() -> dict:
    edges = [timestamp(x) for x in ('2026-06-13', '2026-07-13', '2026-08-13',
                                   '2026-09-13', '2026-10-13', '2026-11-13')]
    return dict(train=[edges[0], edges[3]], wf1_fit=[edges[0], edges[1]], wf1_test=[edges[1], edges[2]],
                wf2_fit=[edges[0], edges[2]], wf2_test=[edges[2], edges[3]],
                validation=[edges[3], edges[4]], holdout=[edges[4], edges[5]],
                full5months=[edges[0], edges[5]])


def catalog() -> list[dict]:
    rows = [(BASELINE, 'baseline', 'trailing', False),
            ('B_sweep_ote', 'sweep_ote', 'trailing', False),
            ('C_deadfish', 'baseline', 'trailing', True),
            ('C_momentum', 'baseline', 'momentum', False),
            ('C_time', 'baseline', 'atr', False),
            ('D_ha_bb_rsi', 'pullback', 'trailing', False),
            ('E_B_deadfish', 'sweep_ote', 'trailing', True),
            ('E_B_pullback', 'sweep_ote_pullback', 'trailing', False),
            ('E_B_pullback_deadfish', 'sweep_ote_pullback', 'trailing', True)]
    return [dict(name=name, variant=variant, exit_mode=mode, deadfish_enabled=deadfish,
                 params={'volume_ratio': 1.2}) for name, variant, mode, deadfish in rows]


def cost_config(arm: dict, scenario: str = 'base') -> CostConfig:
    return CostConfig(stop_atr=2., target_atr=3., trailing_atr=2., cooldown_bars=5,
                      max_hold_bars=30, min_net_rr=1., use_source_exits=False,
                      exit_mode=arm['exit_mode'], deadfish_enabled=arm['deadfish_enabled'],
                      **SCENARIOS[scenario])


def _qualifies(candidate: dict | None, baseline: dict | None) -> bool:
    return bool(candidate and baseline and candidate['N'] >= 100 and candidate['net_pnl'] > 0 and
                candidate.get('profit_factor') is not None and candidate['profit_factor'] > 1 and
                candidate['net_pnl'] > baseline['net_pnl'] and
                candidate['max_drawdown_usdt'] <= baseline['max_drawdown_usdt'])


def choose(training: list[dict], validation: list[dict] | None = None,
           walkforward: list[dict] | None = None) -> dict:
    """Require each individual component AND its combination to help in every evidence set."""
    evidence = [training, *([validation] if validation is not None else []),
                *([walkforward] if walkforward is not None else [])]
    tables = [{row['name']: row for row in rows} for rows in evidence]
    eligible, reasons = [], {}
    for name, components in COMPONENTS.items():
        failed = sorted({component for table in tables for component in [name, *components]
                         if not _qualifies(table.get(component), table.get(BASELINE))})
        reasons[name] = {'qualified': not failed, 'failed_individual_or_combination': failed,
                         'rule': 'N>=100, PF>1, net>0, net>A, dollar MTM DD<=A in each evidence set'}
        if not failed:
            eligible.append(name)
    rank = tables[1] if validation is not None else tables[0]
    winner = min(eligible, key=lambda name: (-rank[name]['net_pnl'], name)) if eligible else BASELINE
    validated = validation is not None and walkforward is not None
    return dict(winner=winner, adopted_components=COMPONENTS.get(winner, []) if validated else [],
                candidate_components=COMPONENTS.get(winner, []), reasons=reasons,
                adoption='research-only validated combination' if eligible and validated else
                'provisional fit candidate; no validation adoption' if eligible else 'A fallback; no component adoption')


def plan(frame: pd.DataFrame, raw: np.ndarray, *, structural: bool = False,
         start: int | None = None, end: int | None = None) -> tuple:
    """Freeze close-based barriers and baseline-cost preview; stress cannot reselect.

    Stops/targets are signal-close mid +/-2/3 closed 5m ATR, then held fixed at
    next open. Optional structural stops can only widen B's common stop.
    """
    if not isinstance(raw, np.ndarray) or raw.dtype != np.int8 or raw.shape != (len(frame),):
        raise ValueError('signals must return int8 ndarray matching feature rows')
    if not np.isin(raw, (-1, 0, 1)).all():
        raise ValueError('signals must contain -1, 0, +1')
    signal = raw.copy()
    stops, targets = np.zeros(len(frame)), np.zeros(len(frame))
    counts = dict(raw_signals=0, invalid_atr=0, invalid_structural_stop=0, preview_min_net_rr=0, preview_pass=0)
    costs = cost_config(catalog()[0])
    for index in np.flatnonzero(raw):
        side, mid, atr = int(raw[index]), float(frame.close.iloc[index]), float(frame.atr.iloc[index])
        counted = (start is None or frame.time.iloc[index] >= start) and (end is None or frame.time.iloc[index] < end)
        counts['raw_signals'] += int(counted)
        if not np.isfinite(atr) or atr <= 0:
            signal[index] = 0
            counts['invalid_atr'] += int(counted)
            continue
        stop, target = mid - side * 2 * atr, mid + side * 3 * atr
        column = 'structural_stop_long' if side == 1 else 'structural_stop_short'
        if structural and column in frame:
            candidate = float(frame[column].iloc[index])
            if not np.isfinite(candidate) or candidate <= 0 or side * (mid - candidate) <= 0:
                signal[index] = 0
                counts['invalid_structural_stop'] += int(counted)
                continue
            stop = min(stop, candidate) if side == 1 else max(stop, candidate)
        stops[index], targets[index] = stop, target
        if net_reward_risk(mid, side, stop, target, costs) < 1.:
            signal[index] = 0
            counts['preview_min_net_rr'] += int(counted)
        else:
            counts['preview_pass'] += int(counted)
    return signal, stops, targets, counts


def _presence(path: Path, *, certified: bool | None = None) -> dict:
    if not path.exists():
        return dict(path=str(path), state='missing', sha256=None)
    state = 'present' if certified is None else 'present_certified' if certified else 'present_uncertified'
    return dict(path=str(path), state=state, sha256=file_hash(path))


def _code(feature_module: object, neuro: Path) -> dict:
    from research_scalping import execution, metrics, data
    paths = [Path(__file__), Path(report.__file__), Path(execution.__file__), Path(metrics.__file__),
             Path(data.__file__), Path(feature_module.__file__), neuro]
    sources = {str(p.resolve()): file_hash(p) for p in paths}
    identities = {}
    compiled = {}

    def code_objects(code):
        yield code
        for constant in code.co_consts:
            if isinstance(constant, CodeType):
                yield from code_objects(constant)

    def stable_code(code):
        # marshal's reference/intern flags can differ for equal imported and
        # freshly compiled code. Hash semantic fields instead of those flags.
        fields = ('co_argcount','co_posonlyargcount','co_kwonlyargcount','co_flags','co_names',
                  'co_varnames','co_freevars','co_cellvars','co_filename','co_firstlineno','co_name')
        return {**{field:getattr(code,field) for field in fields},'bytecode':code.co_code.hex(),
                'constants':[stable_constant(c) for c in code.co_consts]}

    def stable_constant(value):
        if isinstance(value,CodeType):
            return stable_code(value)
        if isinstance(value,frozenset):
            return {'frozenset':sorted((stable_constant(v) for v in value),key=report.canonical)}
        if isinstance(value,tuple):
            return {'tuple':[stable_constant(v) for v in value]}
        return {'type':type(value).__name__,'value':repr(value)}

    functions = {name: getattr(feature_module, name) for name in ('build_features', 'signals')}
    functions.update(execute=run, rr=net_reward_risk, summarize=metrics.summarize,
                     cost_validation=CostConfig.__post_init__)
    for name, function in functions.items():
        if not callable(function) or not hasattr(function, '__code__'):
            raise ValueError('Feature module requires inspectable Python callables')
        filename = function.__code__.co_filename
        path = Path(filename).resolve()
        if filename not in compiled:
            compiled[filename] = set(code_objects(compile(path.read_bytes(),filename,'exec',dont_inherit=True)))
        checksum = digest(stable_code(function.__code__))
        if function.__code__ not in compiled[filename]:
            raise ValueError(f'Loaded callable fingerprint differs from declared source: {name}')
        sources[str(path)] = file_hash(path)
        identities[name] = dict(module=function.__module__, file=str(path),bytecode_sha256=checksum)
    neuro_code = compile(neuro.read_bytes(),str(neuro),'exec',dont_inherit=True)
    return dict(sources=sources, callable_identities=identities,
                neuro_compiled_module_sha256=digest(stable_code(neuro_code)))


def _registration(data_dir: Path, htf_dir: Path | None, neuro: Path, features: object, bounds: dict) -> dict:
    expected = set(period_bounds())
    if set(bounds) != expected or any(
            not isinstance(value, (list, tuple)) or len(value) != 2 or
            any(type(t) is not int or t < 0 or t % 60 for t in value) or value[0] >= value[1]
            for value in bounds.values()):
        raise ValueError('Exact minute-aligned, half-open split bounds required')
    if not (bounds['train'][0] == bounds['wf1_fit'][0] == bounds['wf2_fit'][0] == bounds['full5months'][0] and
            bounds['wf1_fit'][1] == bounds['wf1_test'][0] and
            bounds['wf1_test'][1] == bounds['wf2_fit'][1] == bounds['wf2_test'][0] and
            bounds['wf2_test'][1] == bounds['train'][1] == bounds['validation'][0] and
            bounds['validation'][1] == bounds['holdout'][0] and
            bounds['holdout'][1] == bounds['full5months'][1]):
        raise ValueError('Train, walkforward, validation and holdout chronology required')
    manifest = load_json(data_dir/'manifest.json')
    symbols = [p['symbol'] for p in manifest['pairs']]
    if len(set(symbols)) != len(symbols) or not symbols or any(not re.fullmatch(r'[A-Za-z0-9_-]+', s) for s in symbols):
        raise ValueError('Unique safe configured symbols required')
    inputs = {}
    for pair in manifest['pairs']:
        symbol = pair['symbol']
        candle = _presence(data_dir/f'{symbol}.parquet')
        if pair['status'] == 'complete_candles' and candle['sha256'] != pair.get('parquet_sha256'):
            raise ValueError('Actual parquet fingerprint changed from manifest')
        inputs[symbol] = dict(candles=candle,
            funding=_presence(data_dir/'funding'/f'{symbol}.json', certified=pair.get('funding', {}).get('coverage_certified') is True),
            htf={label: _presence(htf_dir/f'{symbol}_{label}.parquet') if htf_dir else
                 dict(path=None, state='missing', sha256=None) for label in ('1h', '4h')})
    return dict(schema=1, catalog=catalog(), bounds=bounds, code=_code(features, neuro), inputs=inputs,
                runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,
                             pyarrow=importlib.metadata.version('pyarrow')),
                manifest_sha256=file_hash(data_dir/'manifest.json'), manifest=manifest,
                costs={a['name']: {s: asdict(cost_config(a,s)) for s in SCENARIOS} for a in catalog()},
                stress_scope='A and E_selected only; all predeclared arms base',
                selection_rule='each component and combination: pooled N>=100, PF>1, net>0, net>A, DD<=A',
                portfolio_basis='independent 1000-USDT coin sleeves; 10-USDT fixed stop-plus-cost risk; not production shared portfolio',
                quote_evidence='assumed/current spread and tick metadata, not observed historical quotes',
                source_scope='controlled technical ICT / Freqle concept adaptations; no historical Qwen or native Freqtrade reproduction')


def _validate_candles(frame: pd.DataFrame, step: int) -> None:
    names = ['time', 'open', 'high', 'low', 'close', 'volume']
    if not isinstance(frame.index, pd.RangeIndex) or not frame.columns.is_unique or not set(names) <= set(frame):
        raise ValueError('OHLCV with RangeIndex required')
    t = frame.time.to_numpy()
    values = frame[names[1:]].to_numpy()
    if (t.dtype.kind not in 'iu' or (t < 0).any() or (t % step != 0).any() or
            (len(t)>1 and (np.diff(t) != step).any()) or not np.isfinite(values).all() or
            (frame[['open','high','low','close']] <= 0).any().any() or (frame.volume < 0).any() or
            (frame.low > frame[['open','close']].min(axis=1)).any() or
            (frame.high < frame[['open','close']].max(axis=1)).any()):
        raise ValueError('Malformed, noncontiguous or off-grid real OHLCV')


def _load_frame(registration: dict, symbol: str, end: int, features: object, neuro: Path) -> pd.DataFrame:
    item = registration['inputs'][symbol]
    candle = Path(item['candles']['path'])
    if file_hash(candle) != item['candles']['sha256']:
        raise ValueError('Candle bytes changed during phase')
    warm = registration['manifest'].get('warm_start', registration['bounds']['train'][0] - 86400)
    frame = pd.read_parquet(candle, filters=[('time','>=',warm), ('time','<',end)]).reset_index(drop=True)
    _validate_candles(frame, 60)
    if len(frame) != (end-warm)//60 or frame.empty or frame.time.iloc[0] != warm or frame.time.iloc[-1] != end-60:
        raise ValueError('Minute coverage does not match declared exact bounds')
    htf = {}
    for label, step in [('1h',3600),('4h',14400)]:
        evidence = item['htf'][label]
        if evidence['path'] and _presence(Path(evidence['path'])) != evidence:
            raise ValueError('HTF presence/bytes changed during phase')
        if evidence['state'] == 'missing':
            continue
        path = Path(evidence['path'])
        if file_hash(path) != evidence['sha256']:
            raise ValueError('HTF bytes changed during phase')
        bars = pd.read_parquet(path, filters=[('time','<',warm)]).reset_index(drop=True)
        _validate_candles(bars, step)
        if (len(bars) < 200 or (bars.time + step > warm).any() or
                bars.time.iloc[-1] + step != warm // step * step):
            raise ValueError('HTF requires >=200 actually completed prelude bars')
        htf[label] = bars
    if htf and set(htf) != {'1h','4h'}:
        raise ValueError('Both H1/H4 preludes required when supplied')
    original = frame.copy(deep=True)
    result = features.build_features(frame, ict_module_path=neuro, htf_warmup=htf or None)
    if (not isinstance(result,pd.DataFrame) or not result.index.equals(original.index) or
            not REQUIRED <= set(result) or not result[original.columns].equals(original) or not frame.equals(original)):
        raise ValueError('Feature contract changed rows, candles, or required columns')
    if _code(features,neuro) != registration['code']:
        raise ValueError('Source fingerprint changed during feature computation')
    result.attrs['symbol'] = symbol
    return result


def _linked(path: Path, parent: str) -> dict:
    record = verify_seal(path)
    if record.get('registration_sha256') != parent:
        raise ValueError('Seal registration parent fingerprint mismatch')
    return record


def _verify_stage(output: Path, name: str, parent: str) -> dict:
    record = _linked(output/f'{name}_complete.json', parent)
    for relative, checksum in record['files'].items():
        if file_hash(output/relative) != checksum:
            raise ValueError(f'Checkpoint artifact changed: {relative}')
    return record


def _feature_counts(frame: pd.DataFrame, start: int, end: int) -> dict:
    mask = (frame.time >= start) & (frame.time < end)
    return {column:int(frame.loc[mask,column].sum()) for column in REQUIRED
            if column.startswith(('baseline_','sweep_','ote_','fresh_zone_',
                                  'reaction_','pullback_','deadfish_','strong_trend_'))}


def _stage(output: Path, name: str, bounds: list, selected: str, registration: dict,
           features: object, neuro: Path, *, stress: bool = False) -> list[dict]:
    parent = digest(registration)
    complete = output/f'{name}_complete.json'
    if complete.exists():
        _verify_stage(output,name,parent)
        return load_json(output/f'{name}_aggregate.json')
    start, end = bounds
    arms = catalog()
    chosen = next(a for a in arms if a['name'] == selected)
    arms = [*arms, {**chosen, 'name':'E_selected', 'selected_source':selected}]
    pooled, rows, files = {}, [], {}
    for pair in registration['manifest']['pairs']:
        symbol = pair['symbol']
        frame = None
        prepared = {}
        funding_evidence = registration['inputs'][symbol]['funding']
        if _presence(Path(funding_evidence['path']),
                     certified=funding_evidence['state']=='present_certified') != funding_evidence:
            raise ValueError('Funding presence/bytes changed during phase')
        funding = None
        if pair['status'] == 'complete_candles':
            frame = _load_frame(registration,symbol,end,features,neuro)
            if funding_evidence['state'] != 'missing':
                path = Path(funding_evidence['path'])
                if file_hash(path) != funding_evidence['sha256']:
                    raise ValueError('Funding changed during phase')
                funding = [f for f in load_json(path) if start <= f['time'] < end]
        for arm in arms:
            scenarios = SCENARIOS if stress and arm['name'] in (BASELINE,'E_selected') else ['base']
            for scenario in scenarios:
                metadata = dict(coin=symbol, strategy=arm['name'], window=name, scenario=scenario,
                                window_start=start, window_end=end, policy='continuous' if name=='full5months' else 'flat_1000_reset',
                                selected_source=arm.get('selected_source'), funding_state=funding_evidence['state'])
                if frame is None:
                    rows.extend(report.unavailable(metadata,'Configured contract candles unavailable; no alias'))
                    continue
                key = arm['variant']
                if key not in prepared:
                    raw = features.signals(frame,key,arm['params'])
                    prepared[key] = plan(frame,raw,structural=key.startswith('sweep'),start=start,end=end)
                signal, stops, targets, counts = prepared[key]
                opportunity_hash = digest({label: hashlib.sha256(array.tobytes()).hexdigest()
                    for label,array in [('signal',signal),('stops',stops),('targets',targets),('time',frame.time.to_numpy())]})
                destination = output/'runs'/name/symbol/arm['name']/f'{scenario}.json.gz'
                checkpoint = destination.with_name(destination.name+'.checkpoint.json')
                run_key = digest(dict(registration=parent, metadata=metadata, opportunities=opportunity_hash,
                                      costs=asdict(cost_config(arm,scenario))))
                if checkpoint.exists():
                    old = verify_seal(checkpoint)
                    if old['run_key'] != run_key or file_hash(destination) != old['sha256']:
                        raise ValueError('Run checkpoint fingerprint changed')
                    result = load_json(destination)
                else:
                    result = run(frame,signal,cost_config(arm,scenario),start=start,end=end,
                                 funding=funding,stops=stops,targets=targets,record_equity_phases=True)
                    for trade in result['trades']:
                        trade.update(symbol=symbol,strategy=arm['name'],window=name,scenario=scenario)
                    result.update(metadata=metadata,initial_equity=1000.,opportunities_sha256=opportunity_hash,
                                  opportunity_counts=counts,feature_counts=_feature_counts(frame,start,end),
                                  feature_provenance=frame.attrs, funding_evidence={**funding_evidence,
                                  'supplied_settlements':None if funding is None else len(funding),
                                  'window_settlements_sha256':digest(funding),
                                  'coverage_certified':funding_evidence['state']=='present_certified'})
                    atomic_json(destination,result)
                    seal(checkpoint,dict(run_key=run_key,sha256=file_hash(destination)))
                for path in (destination,checkpoint):
                    files[str(path.relative_to(output))] = file_hash(path)
                pooled.setdefault((arm['name'],scenario),[]).append(destination)
                rows.extend(report.groups(result,metadata))
                print(report.canonical(dict(event='durable_run',stage=name,**metadata,
                                             artifact=str(destination.relative_to(output)),
                                             N=len(result['trades']),rejections=result['rejections'])).decode(),flush=True)
        del frame, prepared
    aggregate = [dict(name=arm,window=name,scenario=scenario,**report.combine(paths))
                 for (arm,scenario),paths in sorted(pooled.items())]
    for suffix,value in [('groups',rows),('aggregate',aggregate)]:
        path = output/f'{name}_{suffix}.json'
        atomic_json(path,value)
        files[str(path.relative_to(output))] = file_hash(path)
    path = output/f'{name}_groups.csv'
    atomic_csv(path,rows)
    files[str(path.relative_to(output))] = file_hash(path)
    seal(complete,dict(registration_sha256=parent,files=files,bounds=bounds,selected=selected))
    print(report.canonical(dict(event='durable_stage',stage=name,seal=file_hash(complete))).decode(),flush=True)
    return aggregate


def _base(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row['scenario']=='base' and row['name']!='E_selected']


def _snapshot(output: Path, name: str, rows: list, parent: str) -> dict:
    path = output/f'{name}_selection.json'
    decision = dict(**choose(_base(rows)),registration_sha256=parent,
                    evidence_sha256=digest(rows),fit_window=name)
    if path.exists():
        if _linked(path,parent) != decision:
            raise ValueError('Frozen fit selection changed')
    else:
        seal(path,decision)
        print(report.canonical(dict(event='durable_selection',stage=name,sha256=file_hash(path))).decode(),flush=True)
    return decision


def _wf_evidence(output: Path) -> list[dict]:
    rows = []
    for arm in catalog():
        paths = sorted(path for window in ('wf1_test','wf2_test')
                       for path in (output/'runs'/window).glob(f'*/{arm["name"]}/base.json.gz'))
        rows.append(dict(name=arm['name'], **report.combine(paths)))
    return rows


def _walkforward_policy(output: Path, registration: dict, selection: dict,
                        features: object, neuro: Path) -> None:
    """Execute fit-selected folds with carried cash and mandatory boundary flattening.

    This is a feasible scheduled research policy, distinct from the common
    fixed-capital ablation cells. Bankruptcy parks the sleeve without inventing
    a fresh positive starting balance in the next fold.
    """
    parent = digest(registration)
    if (output/'wfoos_complete.json').exists():
        _verify_stage(output,'wfoos',parent)
        return
    windows = registration['bounds']
    start, end = windows['wf1_test'][0], windows['wf2_test'][1]
    files, rows, paths = {}, [], []
    arms = {a['name']:a for a in catalog()}
    for pair in registration['manifest']['pairs']:
        symbol = pair['symbol']
        evidence = registration['inputs'][symbol]['funding']
        metadata = dict(coin=symbol,strategy='E_selected',window='wfoos',scenario='base',
                        window_start=start,window_end=end,funding_state=evidence['state'],
                        policy='walkforward_carry_cash_force_flat_fold_boundary')
        if pair['status']!='complete_candles':
            rows.extend(report.unavailable(metadata,'Configured contract candles unavailable; no alias'))
            continue
        destination = output/'runs'/'wfoos'/symbol/'E_selected'/'base.json.gz'
        checkpoint = destination.with_name(destination.name+'.checkpoint.json')
        run_key = digest(dict(registration=parent,selection=selection,metadata=metadata))
        if checkpoint.exists():
            saved = verify_seal(checkpoint)
            if saved['run_key']!=run_key or file_hash(destination)!=saved['sha256']:
                raise ValueError('Walkforward checkpoint changed')
            result = load_json(destination)
        else:
            cash, trades, equity, folds, counts, rejections = 1000., [], [], [], {}, {}
            equity_phases = []
            feature_counts, provenance = {}, {}
            if _presence(Path(evidence['path']),certified=evidence['state']=='present_certified') != evidence:
                raise ValueError('Walkforward funding presence/bytes changed')
            funding = None if evidence['state']=='missing' else load_json(Path(evidence['path']))
            for index in (1,2):
                window, fit = f'wf{index}_test',f'wf{index}_fit'
                lo,hi = windows[window]
                source = selection[f'wf{index}_winner']
                if _linked(output/f'{fit}_selection.json',parent)['winner']!=source:
                    raise ValueError('Walkforward source differs from preceding fit seal')
                opening = cash
                frame = _load_frame(registration,symbol,hi,features,neuro)
                arm = arms[source]
                raw = features.signals(frame,arm['variant'],arm['params'])
                signal,stops,targets,opportunities = plan(frame,raw,
                    structural=arm['variant'].startswith('sweep'),start=lo,end=hi)
                provenance[window] = load_json_bytes(frame.attrs)
                for key,value in _feature_counts(frame,lo,hi).items():
                    feature_counts[key] = feature_counts.get(key,0)+value
                if cash>0:
                    part = run(frame,signal,replace(cost_config(arm),initial_equity=cash),start=lo,end=hi,
                        funding=None if funding is None else [f for f in funding if lo<=f['time']<hi],
                        stops=stops,targets=targets,record_equity_phases=True)
                else:
                    part = dict(trades=[],equity=[[lo,cash],[hi,cash]],
                                equity_phases=['initial','bar_close'],rejections={'bankrupt_sleeve':1})
                cash = part['equity'][-1][1]
                for trade in part['trades']:
                    trade.update(symbol=symbol,strategy='E_selected',window='wfoos',scenario='base',selected_source=source)
                trades.extend(part['trades'])
                equity.extend(part['equity'])
                equity_phases.extend(part['equity_phases'])
                for key,value in part['rejections'].items():
                    rejections[key] = rejections.get(key,0)+value
                for key,value in opportunities.items():
                    counts[key] = counts.get(key,0)+value
                folds.append(dict(selection_fit=fit,selected=source,initial_equity=opening,final_equity=cash,
                                  fit_selection_sha256=file_hash(output/f'{fit}_selection.json'),
                                  opportunities_sha256=hashlib.sha256(signal.tobytes()+stops.tobytes()+targets.tobytes()).hexdigest(),
                                  start=lo,end=hi))
                del frame,raw,signal,stops,targets,part
            if not np.isclose(cash,1000.+sum(t['net_pnl'] for t in trades),rtol=1e-10,atol=1e-8):
                raise ArithmeticError('Walkforward carried cash does not reconcile')
            supplied = None if funding is None else [f for f in funding if start<=f['time']<end]
            result = dict(metadata=metadata,initial_equity=1000.,trades=trades,equity=equity,
                          equity_phases=equity_phases,folds=folds,
                          rejections=rejections,opportunity_counts=counts,feature_counts=feature_counts,
                          feature_provenance=provenance,opportunities_sha256=digest([f['opportunities_sha256'] for f in folds]),
                          funding_evidence={**evidence,'supplied_settlements':None if supplied is None else len(supplied),
                                            'window_settlements_sha256':digest(supplied),
                                            'coverage_certified':evidence['state']=='present_certified'})
            atomic_json(destination,result)
            seal(checkpoint,dict(run_key=run_key,sha256=file_hash(destination)))
        rows.extend(report.groups(result,metadata))
        paths.append(destination)
        for path in (destination,checkpoint):
            files[str(path.relative_to(output))] = file_hash(path)
        print(report.canonical(dict(event='durable_run',stage='wfoos',**metadata,
                                   N=len(result['trades']),artifact=str(destination.relative_to(output)))).decode(),flush=True)
    aggregate = [dict(name='E_selected',scenario='base',window='wfoos',**report.combine(paths))]
    for suffix,value in [('groups',rows),('aggregate',aggregate)]:
        path = output/f'wfoos_{suffix}.json'
        atomic_json(path,value)
        files[str(path.relative_to(output))] = file_hash(path)
    path = output/'wfoos_groups.csv'
    atomic_csv(path,rows)
    files[str(path.relative_to(output))] = file_hash(path)
    seal(output/'wfoos_complete.json',dict(registration_sha256=parent,files=files))
    print(report.canonical(dict(event='durable_stage',stage='wfoos',
                               seal=file_hash(output/'wfoos_complete.json'))).decode(),flush=True)


def _report(output: Path, registration: dict, selection: dict) -> dict:
    parent = digest(registration)
    decision = _linked(output/'validation_decision.json',parent)
    stages = ['wf1_fit','wf2_fit','train','wf1_test','wf2_test','wfoos','validation','holdout','full5months']
    for stage in stages:
        _verify_stage(output,stage,parent)
    comparison = []
    for stage in stages:
        for row in load_json(output/f'{stage}_aggregate.json'):
            comparison.append(dict(split=stage,**row))
    result = dict(registration_sha256=parent,selection_sha256=digest(selection),
                  validation_decision_sha256=digest(decision),selected=decision['winner'],live_adoption=False,
                  rejected_variants=decision['reasons'],measurements=comparison,
                  portfolio_basis=registration['portfolio_basis'],source_scope=registration['source_scope'],
                  evidence_class='synthetic engineering fixture; not financial evidence' if
                    registration['manifest'].get('synthetic') is True else 'operator-supplied offline research inputs',
                  invocations=[verify_seal(p) for p in sorted((output/'invocations').glob('*.json'))],
                  continuous='full5months starts once at 1000/coin; carries cash/positions over split boundaries; retrospective diagnostic',
                  reset='train/validation/holdout/WF cells each start flat at 1000/coin; summing returns is not continuous performance',
                  walkforward='wfoos uses preceding fit-only choices, carries cash, force-closes at fold boundary; fixed-capital wf1/2_test ablations remain reset diagnostics',
                  limitations=['Historical Qwen/native Freqtrade reproduction NOT_RUN',
                    'Independent sleeves are not production shared margin/risk portfolio',
                    'Current configured universe has survivorship/selection bias; MATIC never aliased',
                    'Historical quotes/tick specs unavailable; spread/slippage assumed',
                    'Missing/uncertified funding is not known-zero; supplied events only',
                    'Direction drawdown realized-only; month MTM and exit PnL differ',
                    'B structural widening applied only if feature module supplies structural_stop_long/short',
                    'Parent must integrate real features, run real corrected ict_scan parity and AVAX training-only smoke before measurement'])
    atomic_json(output/'report.json',result)
    atomic_csv(output/'comparison.csv',comparison)
    print(report.canonical(dict(event='durable_report',artifact='report.json',sha256=file_hash(output/'report.json'))).decode(),flush=True)
    return result


def execute_phase(data_dir: Path, output: Path, neuro_root: Path, phase: str, *, htf_dir: Path | None = None,
                  workers: int = 2, feature_module: object | None = None, bounds: dict | None = None) -> dict | list:
    """Run stages sequentially, one coin at a time; workers is an upper bound, never a pool multiplier."""
    if type(workers) is not int or workers not in (1,2):
        raise ValueError('workers must be 1 or 2')
    if phase not in ('train','eval','report'):
        raise ValueError('phase must be train, eval, or report')
    neuro = Path(neuro_root)
    if not neuro.is_absolute() or not neuro.is_file() or neuro.suffix != '.py':
        raise ValueError('--neuro-root must declare an absolute corrected ICT .py module path')
    neuro = neuro.resolve()
    features = feature_module or importlib.import_module('research_ict.features')
    data_dir,output = Path(data_dir).resolve(),Path(output).resolve()
    htf_dir = Path(htf_dir).resolve() if htf_dir is not None else None
    registration = _registration(data_dir,htf_dir,neuro.resolve(),features,bounds or period_bounds())
    parent = digest(registration)
    registration_path = output/'registration.json'
    if phase=='train' and not registration_path.exists():
        seal(registration_path,registration)
    elif verify_seal(registration_path) != load_json_bytes(registration):
        raise ValueError('Code/data/funding/HTF/catalog fingerprint changed after freeze')
    invocation = dict(registration_sha256=parent,phase=phase,entrypoint='execute_phase',
                      started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                      executable=sys.executable,parameters=dict(data=str(data_dir),htf=str(htf_dir) if htf_dir else None,
                      neuro_root=str(neuro),output=str(output),workers_upper_bound=workers,actual_workers=1),
                      feature_injected=feature_module is not None)
    index = 1+max((int(p.name.split('_')[0]) for p in (output/'invocations').glob('*.json')),default=0)
    seal(output/'invocations'/f'{index:04d}_{phase}.json',invocation)
    bounds = registration['bounds']
    if phase=='train':
        fit1 = _stage(output,'wf1_fit',bounds['wf1_fit'],BASELINE,registration,features,neuro)
        choice1 = _snapshot(output,'wf1_fit',fit1,parent)
        fit2 = _stage(output,'wf2_fit',bounds['wf2_fit'],BASELINE,registration,features,neuro)
        choice2 = _snapshot(output,'wf2_fit',fit2,parent)
        training = _stage(output,'train',bounds['train'],BASELINE,registration,features,neuro)
        final = _snapshot(output,'train',training,parent)
        selection = dict(registration_sha256=parent,winner=final['winner'],wf1_winner=choice1['winner'],
                         wf2_winner=choice2['winner'],fit_seals={name:file_hash(output/f'{name}_selection.json')
                                                             for name in ('wf1_fit','wf2_fit','train')})
        if (output/'selection.json').exists():
            if _linked(output/'selection.json',parent)!=selection:
                raise ValueError('Frozen selection changed')
        else:
            seal(output/'selection.json',selection)
        return training
    selection = _linked(output/'selection.json',parent)
    for name,checksum in selection['fit_seals'].items():
        if file_hash(output/f'{name}_selection.json')!=checksum:
            raise ValueError('Fit selection changed before test')
        _linked(output/f'{name}_selection.json',parent)
        _verify_stage(output,name,parent)
    if phase=='report':
        return _report(output,registration,selection)
    _stage(output,'wf1_test',bounds['wf1_test'],selection['wf1_winner'],registration,features,neuro)
    _stage(output,'wf2_test',bounds['wf2_test'],selection['wf2_winner'],registration,features,neuro)
    _walkforward_policy(output,registration,selection,features,neuro)
    validation = _stage(output,'validation',bounds['validation'],selection['winner'],registration,features,neuro)
    training = load_json(output/'train_aggregate.json')
    wf = _wf_evidence(output)
    decision = dict(**choose(_base(training),_base(validation),wf),registration_sha256=parent,
                    selection_sha256=digest(selection),evidence_sha256=digest([training,validation,wf]),
                    preceding_stage_seals={name:file_hash(output/f'{name}_complete.json')
                                           for name in ('wf1_test','wf2_test','validation')})
    path = output/'validation_decision.json'
    if path.exists():
        if _linked(path,parent)!=decision:
            raise ValueError('Validation decision changed after seal')
    else:
        seal(path,decision)
        print(report.canonical(dict(event='durable_validation_decision',sha256=file_hash(path),
                                   winner=decision['winner'])).decode(),flush=True)
    # This durable barrier precedes even constructing holdout feature frames.
    if _registration(data_dir,htf_dir,neuro,features,bounds)!=registration:
        raise ValueError('Inputs changed before holdout barrier')
    holdout = _stage(output,'holdout',bounds['holdout'],decision['winner'],registration,features,neuro,stress=True)
    _stage(output,'full5months',bounds['full5months'],decision['winner'],registration,features,neuro,stress=True)
    return holdout


def load_json_bytes(payload: dict) -> dict:
    """Compare through canonical JSON so tuple/list representations cannot drift."""
    import json
    return json.loads(report.canonical(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--htf',type=Path)
    parser.add_argument('--neuro-root',required=True,type=Path,help='Absolute corrected ICT .py module path')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--phase',required=True,choices=('train','eval','report'))
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    args = parser.parse_args()
    execute_phase(args.data,args.output,args.neuro_root,args.phase,htf_dir=args.htf,workers=args.workers)


if __name__=='__main__':
    main()
