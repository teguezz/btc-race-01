// Capitulation-and-regime BTC bot (agent-fun/bars-replay/v1).
//
// Two independent long-only engines share one position slot:
//
// 1) Washout reversal. After a deep drawdown from the trailing six-hour
//    high (close at least 4% below it), the first up bar that closes
//    above the highest close of the prior 15 bars is treated as a
//    capitulation reversal and bought with the full account equity
//    (sizing is capped by the SDK's own weight-room computation). One entry is
//    allowed per washout "episode"; the episode only resets once price
//    recovers to within 1.25% of the trailing high. Exits: a structure
//    stop just below the 30-bar swing low, a +4% target, a trailing stop
//    (arms at +2%, trails 1.2%), a 45-bar time stop if the trade has not
//    gained at least 0.5%, and a 240-bar maximum hold.
//
// 2) Confirmed uptrend. Only when price sits above a rising 300-bar mean
//    with five-hour momentum of at least +1% does a 90-bar breakout open
//    a 90%-of-equity trend position. It is held until price closes 0.5%
//    below the mean (regime loss), a 3% hard stop hits, or 1,200 bars
//    pass; a 60-bar cooldown follows each trend exit.
//
// The race window is 24 hours of one-minute bars (1,440 batches), so no
// new entries are opened in the final 10 bars. Everything is a
// deterministic function of the bars seen so far: rolling sums and
// monotonic deques give O(1) amortized work per bar, and no wall clock,
// randomness, network, or file access is used anywhere.

import { runBars } from './vendor/makefun-bars.ts';
import type { BarsState, Batch, OrderInput } from './vendor/makefun-bars.ts';

const SYMBOL = 'CBBTC';
const PRICE_SCALE = 100000; // 10^5, from this competition's hello
const USDC_RAW = 1000000n; // one whole USDC in quote units
const WINDOW_BARS = 1440; // the 24h race window in one-minute bars

const BUY_CAP_BPS = 100; // entries: refuse fills far above the decision close
const SELL_CAP_BPS = 300; // exits: reliability matters more than the last bps

// Washout-reversal parameters.
const W_HI_W = 360; // trailing-high lookback (bars)
const W_DD_THRESH = 0.04; // required drawdown from that high
const W_EPISODE_RESET = 0.0125; // episode ends this close to the high
const W_TRIG_W = 15; // breakout lookback for the reversal trigger
const W_FRAC = 1.0; // fraction of equity bought (the deepest-conviction setup)
const W_TARGET = 0.04; // take-profit
const W_TRAIL_ARM = 0.02; // trailing stop arms at this gain
const W_TRAIL_PCT = 0.012; // trailing stop distance from the peak close
const W_MAX_HOLD = 240; // bars
const W_SWING_W = 30; // swing-low lookback for the structure stop
const W_SWING_BUF = 0.0015; // stop sits this far below the swing low
const W_TIME_STOP_BARS = 45; // time stop horizon
const W_TIME_STOP_MIN_GAIN = 0.005; // required gain by then

// Uptrend parameters.
const T_SLOW_W = 300; // regime mean lookback
const T_SLOPE_LAG = 90; // the mean must be rising versus this many bars ago
const T_ROC_W = 300; // momentum lookback
const T_ROC_MIN = 0.01; // required momentum
const T_BREAKOUT_W = 90; // breakout lookback for the entry trigger
const T_FRAC = 0.9; // fraction of equity bought
const T_STOP = 0.03; // hard stop
const T_MAX_HOLD = 1200; // bars
const T_COOLDOWN = 60; // bars after a trend exit
const T_REGIME_BUF = 0.005; // regime-loss buffer below the mean

const LATE_BARS = 10; // no new entries in the final bars of the window

// Monotonic deque over a sliding window of values, keyed by bar index.
// With wantMax it tracks the window maximum, otherwise the minimum.
class WindowExtreme {
  private idx: number[] = [];
  private val: number[] = [];
  private head = 0;
  private wantMax: boolean;
  private width: number;
  constructor(wantMax: boolean, width: number) {
    this.wantMax = wantMax;
    this.width = width;
  }
  push(i: number, v: number): void {
    while (this.val.length > this.head) {
      const back = this.val[this.val.length - 1];
      if (this.wantMax ? back <= v : back >= v) {
        this.idx.pop();
        this.val.pop();
      } else break;
    }
    this.idx.push(i);
    this.val.push(v);
    while (this.idx[this.head] <= i - this.width) this.head++;
    if (this.head > 1024) {
      this.idx = this.idx.slice(this.head);
      this.val = this.val.slice(this.head);
      this.head = 0;
    }
  }
  get(): number {
    return this.val[this.head];
  }
}

interface Entry {
  mode: 'bounce' | 'trend';
  price: number; // actual fill price, dollars
  seq: number; // seq of the bar whose open filled the buy
  stopPrice: number; // bounce only: absolute structure-stop price
  peakClose: number;
}

// Bar history and rolling indicators. Array index = arrival order = seq.
const closes: number[] = [];
const opens: number[] = [];
const smaHist: number[] = []; // SMA(T_SLOW_W) per bar, NaN until defined
let smaSum = 0;
const highIncl = new WindowExtreme(true, W_HI_W); // includes current bar
const lowSwing = new WindowExtreme(false, W_SWING_W); // includes current bar
const highTrig = new WindowExtreme(true, W_TRIG_W); // excludes current bar
const highBreak = new WindowExtreme(true, T_BREAKOUT_W); // excludes current bar

// Strategy state.
let entry: Entry | null = null;
let pendingMode: 'bounce' | 'trend' | null = null;
let pendingStop = 0;
let pendingSeq = -1;
let pendingFillPrice = 0;
let episodeUsed = false;
let lastTrendExit = -1000000;

function buyOrder(state: BarsState, frac: number): OrderInput {
  let notional = BigInt(Math.floor(frac * Number(state.equity)));
  const room = state.weightRoom(SYMBOL);
  if (room < notional) notional = room;
  if (state.cash < notional) notional = state.cash;
  if (notional < USDC_RAW) notional = USDC_RAW; // minimum order; cap still applies
  return {
    side: 'buy',
    symbol: SYMBOL,
    notional_quote_units: notional,
    max_slippage_bps: BUY_CAP_BPS,
  };
}

// Decide on the bar just pushed (index i = closes.length - 1). trigHigh
// and breakHigh are the exclusion-window extremes read before the push.
function decide(
  state: BarsState,
  seq: number,
  trigHigh: number,
  breakHigh: number,
): OrderInput[] {
  const i = closes.length - 1;
  const close = closes[i];
  const pos = state.positions.get(SYMBOL);
  const holding = pos !== undefined && pos.qty > 0n;

  // Sync the entry record with the observed position.
  if (holding && entry === null && pos !== undefined) {
    // Prefer the actual fill price; fall back to the position's cost
    // basis (quote units per base unit), then to this bar's close.
    let price = pendingFillPrice;
    if (!(price > 0)) price = (Number(pos.cost) * 100) / Number(pos.qty);
    if (!(price > 0)) price = close;
    entry = {
      mode: pendingMode ?? 'bounce',
      price,
      seq,
      stopPrice: pendingStop,
      peakClose: close,
    };
    pendingMode = null;
    pendingSeq = -1;
    pendingFillPrice = 0;
  }
  if (!holding && entry !== null) entry = null;
  if (!holding && pendingSeq >= 0 && seq > pendingSeq) {
    // The pending buy produced no fill (rejected): forget it.
    pendingMode = null;
    pendingSeq = -1;
  }

  if (holding && entry !== null && pos !== undefined) {
    if (close > entry.peakClose) entry.peakClose = close;
    const held = seq - entry.seq;
    let exit = false;
    if (entry.mode === 'bounce') {
      if (close <= entry.stopPrice) exit = true;
      else if (close >= entry.price * (1 + W_TARGET)) exit = true;
      else if (
        entry.peakClose >= entry.price * (1 + W_TRAIL_ARM) &&
        close <= entry.peakClose * (1 - W_TRAIL_PCT)
      )
        exit = true;
      else if (held >= W_TIME_STOP_BARS && close < entry.price * (1 + W_TIME_STOP_MIN_GAIN))
        exit = true;
      else if (held >= W_MAX_HOLD) exit = true;
    } else {
      if (close <= entry.price * (1 - T_STOP)) exit = true;
      else if (i >= T_SLOW_W - 1 && close < smaHist[i] * (1 - T_REGIME_BUF)) exit = true;
      else if (held >= T_MAX_HOLD) exit = true;
      if (exit) lastTrendExit = seq;
    }
    if (exit) {
      return [
        {
          side: 'sell',
          symbol: SYMBOL,
          qty: pos.qty,
          max_slippage_bps: SELL_CAP_BPS,
        },
      ];
    }
    return [];
  }

  // Flat: no entries during warm-up or in the final bars.
  if (state.warmingUp || seq >= WINDOW_BARS - LATE_BARS) return [];

  // --- Washout reversal ---
  if (i >= W_HI_W) {
    const dd = close / highIncl.get() - 1;
    if (dd >= -W_EPISODE_RESET) episodeUsed = false;
    if (!episodeUsed && dd <= -W_DD_THRESH && close > trigHigh && close > opens[i]) {
      episodeUsed = true;
      pendingMode = 'bounce';
      pendingStop = lowSwing.get() * (1 - W_SWING_BUF);
      pendingSeq = seq;
      return [buyOrder(state, W_FRAC)];
    }
  }

  // --- Confirmed uptrend ---
  if (i >= T_SLOW_W + T_SLOPE_LAG && seq - lastTrendExit >= T_COOLDOWN) {
    const slow = smaHist[i];
    const slowPrev = smaHist[i - T_SLOPE_LAG];
    const roc = close / closes[i - T_ROC_W] - 1;
    if (close > slow && slow > slowPrev && roc >= T_ROC_MIN && close > breakHigh) {
      pendingMode = 'trend';
      pendingStop = 0;
      pendingSeq = seq;
      return [buyOrder(state, T_FRAC)];
    }
  }

  return [];
}

await runBars({
  onBatch(state: BarsState, batch: Batch) {
    for (const f of batch.fills) {
      if (f.side === 'buy' && f.symbol === SYMBOL && f.status === 'filled') {
        pendingFillPrice = Number(BigInt(f.price)) / PRICE_SCALE;
      }
    }
    let acted: OrderInput[] = [];
    for (const bar of batch.bars) {
      if (bar.symbol !== SYMBOL) continue;
      const c = Number(BigInt(bar.c)) / PRICE_SCALE;
      const o = Number(BigInt(bar.o)) / PRICE_SCALE;
      const l = Number(BigInt(bar.l)) / PRICE_SCALE;
      // Exclusion windows are read before the current bar is pushed.
      const trigHigh = highTrig.get();
      const breakHigh = highBreak.get();
      opens.push(o);
      closes.push(c);
      const i = closes.length - 1;
      smaSum += c;
      if (i >= T_SLOW_W) smaSum -= closes[i - T_SLOW_W];
      smaHist.push(i >= T_SLOW_W - 1 ? smaSum / T_SLOW_W : NaN);
      highIncl.push(i, c);
      lowSwing.push(i, l);
      acted = decide(state, batch.seq, trigHigh, breakHigh);
      highTrig.push(i, c);
      highBreak.push(i, c);
    }
    return acted;
  },
});
