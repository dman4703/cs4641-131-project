# Data Preprocessing Plan (tick → NBBO-join → bars → features → labels → CV)

## A) Scope & artifacts

* **Datasets (immutable → derived):**

  1. **Raw CSVs** (your vendor export; read-only)
  2. **Cleaned ticks** (trades with attached NBBO; no bars)
  3. **Preprocessed bars** (features + labels; CV splits)
* **Partitioning:** `ticker/date` for both cleaned ticks and bar data.
* **Metadata:** a master Parquet/SQLite metadata table recording config (filters, bar parameters, feature windows), plus summary stats and removal counts for each (ticker, day).

---

## B) Ingest & canonical schema (before any filtering)

1. **Header & schema fix:** drop junk first line, enforce a single header.
2. **Column names:** `ts, type ∈ {TRADE, BEST_BID, BEST_ASK}, price, size, cond, exch, ticker`.
3. **Dtypes:** numeric for `price/size`, categorical for `type/exch`, string for `cond`.
4. **Timezone:** localize to `America/New_York`, then convert to UTC for storage/modeling.
5. **Session clip:** confirm first/last 5 min excluded; re-enforce if needed.
6. **Sort & de-dup:** sort by (`ts`,`type`,`exch`,`price`,`size`); de-dup on `["ts","type","price","size","exch","cond"]` (keep last).
7. **Basic validation:** drop nulls; flag `price≤0`, `size≤0`. Log and remove.

*Rationale:* all downstream logic assumes a clean, typed, time-ordered tape. 

---

## C) Split streams & build NBBO (quote quality first)

1. **Split:** `trades = type==TRADE`; `quotes = type∈{BEST_BID,BEST_ASK}`.
2. **Pivot to NBBO:** produce `nbb, nbo, nbb_size, nbo_size` time series; forward-fill **within day**.
3. **Sanity checks:** drop/repair **locked/crossed** quotes (`nbo < nbb`). Decide policy: carry last valid NBBO or mark as invalid and exclude affected trades from features/labels.
4. **Forward-fill horizon:** cap ffill (e.g., 1–2 seconds) to avoid stale quotes; beyond cap → mark missing.
5. **Compute mid/spread:** `mid = 0.5*(nbb+nbo)`, `spread = nbo-nbb`. Keep both.
6. **Exchange policy:** use consolidated NBBO; if you ever filter exchanges, document the list and reason.

*Rationale:* feature integrity depends on valid NBBO; crossed/locked handling and ffill policy must be explicit.  

---

## D) Attach NBBO to trades (as-of join)

* **Join:** `merge_asof` (direction=`backward`) from trades to NBBO on `ts`.
* **Coverage target:** >99% trades get a valid `mid/spread`; track and investigate shortfalls (e.g., trades before first quote).
* **Microstructure flags (useful later):** `at_bid`, `at_ask` via tolerance to `nbb/nbo`.

*Rationale:* this preserves causality (quote must precede trade). 

---

## E) Condition codes (explicit whitelist, logged)

* **Whitelist “eligible last sale”** conditions; exclude late, out-of-sequence, corrected, averaged, bunched, stopped, derivatively priced, etc.
* **Audit trail:** for each file, log counts removed by reason and persist a sidecar CSV.

*Rationale:* don’t hand-wave “irregular trades”—make it reproducible.  

Short version: in US equity ticks, **condition codes** (aka “sale conditions”) explain *what kind of trade it was* (regular, opening/closing print, odd lot, ISO, after-hours, etc.), and **exchange/market-center codes** explain *where it printed* (NYSE, Nasdaq, Cboe venues, IEX, MEMX, etc.). Bloomberg largely relays the SIP (CTA/UTP) standards and also uses its own two-letter venue codes; both mappings are below with sources.

### 1) Trade “condition” (sale) codes — what they mean

These come from the SIP specs. Multiple codes can appear on one trade (priority rules apply). The most common ones you’ll see:

| Code           | Meaning (CTA/UTP)                                                                     |
| -------------- | ------------------------------------------------------------------------------------- |
| (blank) or `0` | Regular trade (consolidated tape eligible).           |
| `T` / `U`      | Form-T (reported late) / Extended-hours trade.        |
| `O` / `Q`      | Official opening / opening price (market-center open). |
| `6` / `M`      | Closing print / market-center official close.        |
| `5`            | Re-opening print.                                     |
| `L`            | Sold last (late report).                              |
| `I`            | Odd lot trade.                                        |
| `F`            | ISO (Intermarket Sweep Order).                       |
| `H`            | Price-variation trade (outside reference bands).        |
| `4`            | Derivatively priced.                                  |
| `W` / `B`      | Average-price / Bunched trade.                       |
| `X`            | Cross trade.                                          |
| `S`            | Split trade.                                         |
| `7`            | Exempt Qualified Contingent Trade (QCT).                |
| `N`            | Next-day settlement.                                   |
| `P` / `R`      | Prior-reference price / Seller’s option.              |

> Tip for cleaning: for intraday backtests, practitioners typically **keep only “regular” prints during core hours** and **drop** odd lots, after-hours (`T`/`U`), opening/closing prints (`O`,`Q`,`5`,`6`,`M`), derivatively priced (`4`), average/bunched (`W`,`B`), and late/sold-last (`L`) — unless your methodology explicitly needs them. (This selection logic follows the SIP semantics above.

### 2) Exchange / market-center codes — where it printed

There are **two common code sets** you’ll encounter in Bloomberg-sourced data:

#### A) SIP “Participant ID” (1-character) market-center codes

These are the standard venue letters on SIP feeds (and many normalized datasets):

| Code      | Venue                               |
| --------- | ----------------------------------- |
| `N`       | NYSE                                |
| `P`       | NYSE Arca                           |
| `A`       | NYSE American                       |
| `M`       | NYSE Chicago                        |
| `Q` / `T` | Nasdaq (Tape C / Tape A/B contexts) |
| `B`       | Nasdaq BX                           |
| `X`       | Nasdaq PSX                          |
| `Z`       | Cboe BZX                            |
| `Y`       | Cboe BYX                            |
| `K`       | Cboe EDGX                           |
| `J`       | Cboe EDGA                           |
| `V`       | IEX                                 |
| `U`       | MEMX                                |
| `D`       | FINRA ADF (quotes)                  |

#### B) **Bloomberg** two-letter **exchange codes**

Bloomberg’s security master and tick services often label venues with a **two-letter code** (e.g., `UN`, `UP`, `UW`). A reliable public mapping is here (includes corresponding MICs):

| Bloomberg   | Venue (MIC)                  |
| ----------- | ---------------------------- |
| `UN`        | NYSE (XNYS)                  |
| `UP`        | NYSE Arca (ARCX)             |
| `UM`        | NYSE Chicago (XCHI)          |
| `UW`        | Nasdaq Global Select (XNAS)  |
| `UQ`        | Nasdaq Global Market (XNAS)  |
| `UR`        | Nasdaq Capital Market (XNAS) |
| `UX`        | Nasdaq PSX (XNAS)            |
| `VF`        | IEX (IEXG)                   |
| `VG`        | MEMX (MEMX/MEMX U.S.)        |
| `VJ`        | Cboe EDGA (EDGA)             |
| `VK`        | Cboe EDGX (EDGX)             |
| `UU` / `UV` | OTC venues (XOTC/Other OTC)  |

> Note: Bloomberg also exposes **country codes in tickers** (e.g., “AAPL US Equity”) which are not the same as **exchange codes**; use the exchange field in your data (or the mapping above) when you need the precise venue.

### If your Bloomberg file shows unfamiliar codes

* For **sale/condition codes** that don’t match the one-byte SIP list above: Bloomberg sometimes surfaces **comma-separated condition tags** that map from venue-specific flags (e.g., ISO) to the SIP’s standardized set. Use the SIP table above to normalize and filter.
* For **exchange codes**, decode with either the **SIP Participant ID** (1 character) or Bloomberg’s **2-letter code** using the mapping source above.

---

## F) Column cleanup

* **Remove** vendor `TradeTime`/`Spread` columns **only after** deriving NBBO-based `mid/spread`.
* Retain: `ts, ticker, price, size, cond, exch, nbb, nbo, mid, spread, at_bid, at_ask`.

*Rationale:* avoid losing information you still need to compute. 

---

## G) Persist “cleaned ticks”

* **Write Parquet** (pyarrow, zstd), partition `ticker/date`.
* **Quality gates (fail fast):**

  * Monotone `ts` post-sort ✅
  * Duplicate removal rate logged ✅
  * Crossed quotes <0.5% of quote stamps ✅
  * NBBO coverage >99% ✅
  * NaNs after join = 0 ✅

*Rationale:* this is your stable foundation for all bar/feature/label work. 

---

## H) Bar construction (decide empirically)

* **Candidates:** **Dollar** bars and **Volume** bars (optionally benchmark Tick/Imbalance later).
* **Decision metrics:**

  * Coefficient of variation of inter-bar **duration** (lower is better),
  * |ACF(1)| of bar returns (lower is better),
  * Stationarity diagnostics (ADF/KPSS) on returns.
* **Pick winner per ticker** (or globally) and record parameters in metadata table.

*Rationale:* don’t guess—measure. Dollar bars often compare better across price levels; confirm with your data.  

---

## I) Feature engineering (on the bar series)

* **From trades/NBBO aggregated to bars:**

  * `vwap`; **VWAP distance** and its **z-score**;
  * **Bollinger position** (price vs BB bands on mid or bar close);
  * **Short-term momentum** (returns over 1–5 bar/window options);
  * **Relative volume** vs intraday profile;
  * **Time-of-day** encoding (sin/cos of minutes since open);
  * **Microstructure**: effective spread, fraction at bid/ask, simple order-flow imbalance.
* **Parameterize** windows and keep them in metadata table.

*Rationale:* features should be based on actual prices/volumes, then standardized. 

---

## J) Standardization / normalization (after features; avoid look-ahead)

* **Granularity:** per **ticker-session** (i.e., per day) to avoid cross-day leakage.
* **Choice:** either (a) plain per-day mean/std, or (b) **expanding-window**/intraday-profile for live-simulation realism.
* **Persist stats:** store `{ticker, date, feature: mean/std or profile}` for reproducibility.

*Rationale:* standardize **after** feature engineering; never before NBBO/microstructure computation. 

---

## K) Labeling (triple barrier)

* **Vol scaling:** set upper/lower barriers as ±k·σ using EWMA or rolling realized vol; **time barrier** (e.g., 15–30 min).
* **Event record:** start time, barrier levels, touch order, outcome `{tp, sl, time}`.
* **Note:** GMM doesn’t need labels; your tree/boosting models do.

*Rationale:* volatility-scaled targets are comparably meaningful intraday. 

---

## L) Cross-validation (purged K-fold with embargo)

* **Splitting unit:** day (chronological; no shuffle).
* **Purge:** remove overlapping label horizons between train and test.
* **Embargo:** ≥ max holding period (e.g., 30–45 min) after each test fold’s end.

*Rationale:* this is the standard to eliminate leakage from overlapping events. 

---

## M) Persist “preprocessed bars”

* **Write Parquet** (pyarrow, zstd), partition by `ticker/date/bar_type`.
* **Artifacts:** bars + features + labels + fold assignment + metadata table.
* **Quality gates:** zero NaNs, consistent feature windows, barrier coverage rates logged.

---

## N) Defaults & decisions (set now; tune later)

* **FFill cap for quotes:** 1–2s.
* **Cond-code whitelist:** vendor-specific “eligible last sale”; log every exclusion reason.
* **Bar choice:** start with **Dollar bars**, validate against Volume via metrics above.
* **Feature windows:** momentum (3/5/10 bars), BB (20 bars), rel-vol (intraday profile).
* **Triple barrier:** k ∈ {1.0, 1.5}, time barrier 20 min baseline.
* **Embargo:** 30–45 min.
* **Compression/types:** Parquet + zstd; `float32` for prices/spreads; `int32` for size; categories for `type/exch`.

*Everything is recorded to metadata for reproducibility.*  

---

## O) Run-time checks (per file/day)

* Monotone `ts` ✔︎
* Duplicate-removal % ✔︎
* Crossed-quote % ✔︎
* NBBO coverage % ✔︎
* Feature NaNs ✔︎
* Label coverage (triple-barrier) ✔︎
* CV split leakage checks ✔︎

*Fail fast if any gate trips.* 

---

### Bottom line

* Standardize **after** features (not before NBBO/feature calc).
* Be explicit and logged about **cond-code filtering** and **quote sanity**.
* **Choose bars empirically**, document every parameter, and persist per-day stats/metadata for reproducibility.

If you want, tell me your cond-code dictionary and intended forward-fill cap; I’ll plug in concrete defaults consistent with your vendor.  
