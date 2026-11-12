# Five-indicator YouTube strategy — 2026-08-12 to 2026-11-12

Source: https://youtu.be/d5tsDoX7U3w (Turkish; this is an independent
mathematical reconstruction, not a copy of the original indicators' code).

## Exact values transcribed from the video

- RSI Bands [LazyBear]: OB 70, OS 30, length 15; middle=(upper+lower)/2.
- Two-Pole Oscillator [BigBeluga]: filter length 12; zero-line buy/sell dots.
- Stochastic RSI: K 3, D 3 (D hidden), RSI length 8, Stochastic length 10,
  Source Close, display levels 0/100.
- Supertrend: ATR 10, hl2, multiplier 1.2, Wilder ATR.
- EMA: 200 Close.
- Indicators use HEIKIN ASHI; execution uses REAL OHLCV.

## Backtest assumptions not specified in the video

- 24-bar RSI-band setup window; 12-bar two-pole / Stoch / Supertrend
  confirmation window; Stoch K at least 98 for long or at most 2 for short.
- Use only confirmed candles: BigBeluga's dot is drawn one bar to the left,
  but is not actionable until the confirmation bar.
- Enter at the NEXT bar open, never at the signal bar's already-known close.
- On each bar, touch of RSI middle band requests 50% reduction, touch of the
  opposite band requests full exit, and stop requests full exit. Execution is
  at the following real open (not at touched price); stop wins tie.
- Stop beyond RSI band by 0.20% with a hard cap at 10% margin loss at 10x
  (about 1% adverse price movement before costs).
- 10% equity as margin per signal; 10x cross approximate, 1000 USDT starting
  capital, 0.05% taker fee on both entry and exit (conservative market-order approximation), 0.03% slippage per fill.
- Historical funding, mark-price liquidation and intrabar stop/TP price
  fills are NOT modeled. The resulting returns are exploratory, not
  executable-performance claims.

## Reproduce

From repo root:

    pip install -e .
    cd agent
    python -m experiments.video_five_indicators.run --timeframe 1H

This downloads BTCUSDT, ETHUSDT, SOLUSDT futures OHLCV from official public
Binance Vision monthly and daily ZIP archives, including a month of indicator
warmup (July 1–Aug 11). The test window is Aug 12 00:00 UTC to Nov 12 00:00
UTC exclusive (all of Nov 11). Every expected candle must be present or the
run FAILS CLOSED; no synthetic data is substituted. Each archive SHA256 is
recorded. Raw prices are not committed. Run artifacts and complete metric
JSON are saved under video-five-backtest-results.

Run offline unit checks with:

    PYTHONPATH=agent pytest agent/tests/test_video_five_indicators.py -q

A dedicated GitHub Actions workflow also executes and uploads the evidence.

15m comparison uses identical dates and strategy parameters but 15-minute bars; all results are stored as per-timeframe Actions artifacts. The tests are exploratory only: indicators are independently reconstructed, and strict perpetual mark/funding/liquidation is not modeled.
