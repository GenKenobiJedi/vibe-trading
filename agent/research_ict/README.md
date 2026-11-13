# Offline ICT / Freqle concept research

This fork-only source module is an **offline experiment**, not a scanner, broker
connector or live strategy. It never calls a model or publishes trading signals.
Its signal generation, accounting and phase driver consume supplied local data.
Optional `warmup` / `funding` collectors use public BingX GET requests only.

## Scope and source differences

- A: corrected technical ICT on completed H1/H4 prefixes, plus completed 5m
  confirmation, a matching eligible ICT zone and 1m reaction.
- B: trend -> already-known swing sweep -> subsequent close-confirmed structure
  -> frozen 62–79% OTE -> later first tap of an overlapping fresh FVG/OB ->
  reaction/volume. An unrelated later zone cannot authorize entry; a failed zone
  cannot overwrite another valid same-candle zone. LONG/SHORT are mirrored.
- C: identical baseline opportunities with trailing, guarded Deadfish, momentum
  or time exits. Deadfish requires joint volume/BB/momentum weakness, insufficient
  progress, minimum hold and no strong-trend protection. Fading momentum need not
  wait for EMA ordering to reverse.
- D: recursive Heikin-Ashi / BB40 / RSI confirmation **in addition** to the ICT
  trend/zone entry, never an independent countertrend trade.
- E: predeclared combinations. Each constituent and its combination must pass
  net profitability, PF, minimum pooled trade count and baseline-relative dollar
  drawdown conditions in train, walk-forward tests and validation. Otherwise
  `E_selected` explicitly falls back to A; no improvement is claimed.

Concept references: [LiquiditySweep](https://freqle.org/strategy/1f9ed9468ca38ef9),
[DS_BNC_EV1E_5m](https://freqle.org/strategy/d8038a9c2901a501),
[ClucHAnix1m](https://freqle.org/strategy/0951cd9cace5d663).
The requested OTE, reaction/volume gates, short side, momentum and RSI details
are deliberate adaptations, not original-strategy reproduction. Public Freqle
league numbers are not evidence for these BingX experiments.

## Inputs and invocation

Use a source checkout and an isolated Python environment with NumPy, pandas,
PyArrow, TA-Lib, requests and pytest. Do not use a production environment or copy
credentials. Supply the trusted, corrected `ict_analysis.py` explicitly; loading
this Python module executes operator-selected local code, not downloaded source.

`DATA/manifest.json` declares integer UTC `warm_start` and `pairs`, including all
unavailable symbols. Each complete pair has `symbol`, `status: complete_candles`
and `parquet_sha256`; `DATA/SYMBOL.parquet` contains contiguous real minute opens
and `time,open,high,low,close,volume`. Funding is optional
`DATA/funding/SYMBOL.json`, a list of `time,rate,mark_price` records. The manifest's
`funding.coverage_certified` is true only with independent coverage evidence;
present-but-uncertified or missing funding is not known zero. Supply genuine
`HTF/SYMBOL_1h.parquet` and `SYMBOL_4h.parquet` prelude bars ending before minute
warmup, at least 200 completed bars of each scale. No filling or symbol aliases.

The default fixed study is UTC `[2026-06-13,2026-11-13)`: train ends September 13,
validation ends October 13, final holdout ends November 13. Two expanding fits
end July 13 and August 13 and precede their respective month-long tests.
These dates are the commissioned study, not an implicit generic date interface.

```bash
export PYTHONPATH="$PWD/agent"
python -m research_ict.runner --help
python -m research_ict.runner --data "$DATA" --htf "$HTF" \
  --neuro-root "$ICT_MODULE" --output "$OUT" --phase train --workers 2
python -m research_ict.runner --data "$DATA" --htf "$HTF" \
  --neuro-root "$ICT_MODULE" --output "$OUT" --phase eval --workers 2
python -m research_ict.runner --data "$DATA" --htf "$HTF" \
  --neuro-root "$ICT_MODULE" --output "$OUT" --phase report --workers 2
```

`ICT_MODULE` must be an absolute **file** path. One coin is processed at a time;
`workers` is an upper bound (1 or 2), not a promise of parallel execution. Use a
new output directory for changed code/data, and preserve old evidence. Resume
checks frozen file/callable/data hashes and durable run/stage seals. Do not run
two writers against one experiment. Nothing authorizes orders or deployment.

## Execution, artifacts and limits

All arms share 10x linear futures, independent 1,000-USDT coin sleeves, maximum
planned 10-USDT stop-plus-cost risk, five-bar cooldown and 30-bar hold cap. Initial
barriers are 2/3 closed 5m ATR for stop/target; B's supplied structure may only
widen the stop. Signals close first, execute next open in real prices and check
net RR again. Same-candle unknown ordering is stop-first; close-derived trailing
ratchets apply next candle. Fees/funding are charged once; adverse spread/slippage
are already in fills. Gaps may exceed planned risk. No margin-driven silent size
clamp. Base fee is 0.0005/leg, slippage 0.0002/leg, full spread 2 bps. Additional
fee/slippage/combined scenarios retain frozen opportunities; A and E receive
holdout and full-period stress, all arms receive base-cost measurements.

The driver seals the final validation decision **before constructing holdout
features**. Flat-reset training/fold/validation/holdout cells, carried-cash
walk-forward policy and continuous full-period diagnostics are separate. The
full-period E choice is retrospective, not deployable early-period OOS evidence.

Artifacts include sealed registration and selections, `validation_decision.json`,
compressed trade/equity ledgers under `runs/`, stage group CSV/JSON by
coin/month/direction, and final `report.json` / `comparison.csv`. Monthly MTM and
whole-trade exit-month PnL are distinct. Optional execution-phase evidence
separates the prior completed candle from next-open/funding events sharing a UTC
midnight timestamp; ambiguous untagged boundaries are rejected. Bar-close proxy
exits are attributed to the completed candle, open exits to the new interval.
Side drawdown is realized-only; aggregate
close-equity DD cannot synchronize independent sleeves' intrabar extrema. No-trade
WR/PF are null, unavailable symbols are `NOT_EVALUABLE` throughout the matrix.

Historical bid/ask/tick metadata, a fully certified funding schedule, historical
universe membership and production shared-margin admission are not established
by this module. Current-constituent selection has survivor bias. This is neither
historical Qwen replay nor native Freqtrade or Vibe `CryptoEngine` fill parity.
Do not adopt a component from unit tests, an empty result or assumed-cost profits.
No production component is enabled automatically.

```bash
PYTHONPATH=agent python -m pytest \
  agent/tests/research_scalping agent/tests/research_ict -q --tb=short
python tools/ci_private_files_gate.py
git diff --check
```

Keep datasets, run outputs, generated reports and local `docs/` notes outside
Git. Publish only sanitized fixtures/source and preserve operator financial
exports separately. This README is a method description, not a profitability or
independent-review claim.
