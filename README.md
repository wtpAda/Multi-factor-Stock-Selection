# Index Enhance and Stock Selection

An adaptive, rolling-window multi-factor research framework for S&P 500 stock selection and index enhancement.

This project is built as a full research pipeline, not just a single backtest script. It covers factor engineering, IC-based factor selection, monthly signal generation, execution timing, risk-aware portfolio construction, transaction-cost modeling, and benchmark comparison.

## What This Codebase Actually Does

At a high level, the system answers one practical question:

> "Given only information available up to each month-end, which stocks should be long/short next month, and how should positions be weighted under risk and cost constraints?"

It does this in six connected layers:

1. Data Layer: load and align multi-ticker daily OHLCV.
2. Factor Layer: compute trend and momentum/reversal features per ticker.
3. Selection Layer: estimate cross-sectional IC and keep the strongest factors.
4. Signal Layer: convert factor values into long/short/hold using IC-weighted voting.
5. Portfolio Layer: apply monthly execution timing, risk parity weights, SL/TP, cooldown, and trading costs.
6. Analytics Layer: output NAV, drawdown, risk metrics, diagnostics, and SPX comparison.

## Core Modules

| Module | Role |
|---|---|
| `Functions_v2.py` | Core engine: factor generation, IC estimation, signal generation, and performance calculation |
| `Backtesting_v2.py` | End-to-end rolling experiment orchestration |
| `get_rf_data.py` | Optional downloader for daily risk-free rate from FRED (`DTB3`) |
| `SPX_data_prep.py` | Optional helper for local data collection/prep |

## Factor Engineering Library

The framework currently combines two factor families.

### 1) MA Structure Factors (`MA_Area`)

Designed to capture moving-average geometry and trend state transitions:

- `MaAreaSM`: short MA minus mid MA
- `MaAreaML`: mid MA minus long MA
- `MaBreakSM`: short/mid upward vs downward crossover indicator
- `MaBreakML`: mid/long upward vs downward crossover indicator
- `MaDurationSM`: relative up/down state of short vs mid MA
- `MaDurationML`: relative up/down state of mid vs long MA

These features encode trend strength, inflection, and persistence.

### 2) Momentum/Reversal Factors (`momentum_reverse`)

Designed to decompose return behavior across horizons and micro-structures:

- Directional momentum:
`mmt_normal_M`, `mmt_normal_A`
- Mean-relative momentum:
`mmt_avg_M`, `mmt_avg_A`
- Intraday/overnight decomposition:
`mmt_intraday_M`, `mmt_intraday_A`, `mmt_overnight_M`, `mmt_overnight_A`
- Volatility-state-conditioned return spread:
`mmt_range_M`, `mmt_range_A`
- Path efficiency style signal:
`mmt_route_M`, `mmt_route_A`

The default implementation yields 18 factors in total (6 MA + 12 momentum/reversal).

## Rolling Research Protocol

The backtest uses a monthly rolling training regime:

1. Select a month-end `T`.
2. Build a trailing training window of `ROLLING_MONTHS` ending at `T`.
3. Generate daily factors inside that window.
4. Robustify factors via outlier clipping (median-centered bounds).
5. Standardize factors cross-sectionally by date.
6. Build forward-return labels (`FORWARD_RETURN_PERIOD` days).
7. Estimate daily cross-sectional Pearson IC by factor.
8. Rank factors by absolute mean IC and keep Top-N.
9. Aggregate factor values to month-end and create signals at `T`.
10. Shift signals to next month for execution.

This design keeps model selection adaptive while maintaining strict month-end decision timing.

## Signal Decision Engine

Signal generation is not a simple threshold on one factor. It is a weighted voting system:

- For each selected factor, assign direction from sign(mean IC).
- Build factor-specific long/short triggers from empirical monthly quantiles.
- For each stock, each factor contributes to buy/sell score only if:
  - factor value passes trigger
  - factor IC weight exceeds threshold
- Final action:
  - `long` if buy score > sell score
  - `short` if sell score > buy score
  - `hold` otherwise

This allows dynamic regime response without hard-coding a fixed factor polarity.

## Portfolio Construction and Risk Controls

Execution and risk logic in `calculate_performance(...)` includes:

- Signal timing shift: month-end signals become active at next month start.
- Position mapping: `long` uses +return, `short` uses -return.
- Weighting: inverse-volatility allocation on active names (risk parity style).
- Capital mode:
  - `fully_invested=True`: normalize active weights to full deployment
  - `fully_invested=False`: allow cash drag when few names are active
- Risk exits:
  - stop-loss (`SL_THRESHOLD`)
  - take-profit (`TP_THRESHOLD`)
  - optional cooldown after exit within same month
- Trading friction:
  - turnover-based transaction cost (`transaction_cost_bps`)
  - optional charge only on rebalance days (`cost_on_rebalance_only`)

## Performance and Diagnostics

The framework reports both strategy and benchmark context:

- Annualized return
- Annualized volatility
- Sharpe ratio
- Max drawdown
- Historical daily VaR(95%)
- RF-adjusted Sharpe and Sortino (if `risk_free_rate.csv` is provided)
- Signal distribution diagnostics
- Active-position breadth diagnostics
- Strategy beta/correlation vs market proxy
- Strategy vs SPX comparison table and normalized NAV chart

Generated files:

- `rolling_strategy_nav.png`
- `rolling_strategy_drawdown.png`
- `rolling_strategy_vs_spx.png` (if SPX data is available)

## Anti-Lookahead Design

The code includes explicit anti-leakage controls:

- Forward labels can be hard-cut with `cutoff_date`.
- IC estimation is computed only inside each training window.
- Live-month execution uses shifted month-end signals (no same-day look-ahead).

## Local Input Files (Not Stored in Repo)

Place local files in repository root before running:

- `combined_sp500_data.csv` (required)
- `spx_data.csv` (optional)
- `risk_free_rate.csv` (optional)

Expected schema:

| File | Required Columns |
|---|---|
| `combined_sp500_data.csv` | `Ticker`, `Date`, `Open`, `High`, `Low`, `Close`, `Volume` |
| `spx_data.csv` | `Date`, and either `Close` or `Adj Close` |
| `risk_free_rate.csv` | `Date`, `rf_daily` |

## Setup and Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run rolling backtest:

```bash
python Backtesting_v2.py
```

## Key Configuration (`Backtesting_v2.py`)

| Parameter | Default | Description |
|---|---:|---|
| `ROLLING_MONTHS` | `5` | Number of months in each rolling training window |
| `N_TOP_FACTORS` | `3` | Number of factors selected by IC |
| `FORWARD_RETURN_PERIOD` | `21` | Label horizon in trading days |
| `VOLATILITY_LOOKBACK` | `30` | Lookback for inverse-volatility estimation |
| `SL_THRESHOLD` | `0.05` | Stop-loss trigger |
| `TP_THRESHOLD` | `0.30` | Take-profit trigger |
| `IC_WEIGHT_THRESHOLD_FOR_SIGNAL` | `0.02` | Minimum IC weight for factor vote |
| `LONG_TRIGGER_Q` | `0.8` | Long trigger quantile |
| `SHORT_TRIGGER_Q` | `0.2` | Short trigger quantile |
| `LONG_TRIGGER_FALLBACK` | `0.15` | Fallback long trigger when quantile is missing |
| `SHORT_TRIGGER_FALLBACK` | `-0.15` | Fallback short trigger when quantile is missing |

## Data Policy

This repository is intentionally code-only.  
Do not upload raw data CSV files to GitHub.

If data files were tracked previously, untrack them while keeping local copies:

```bash
git rm -r --cached combined_sp500_data.csv spx_data.csv SPX.csv sp500_data
git add .gitignore
git commit -m "Stop tracking local market data files"
```

## Scope and Disclaimer

This repository is a research framework for factor validation and strategy prototyping.  
It is not production trading infrastructure.

For production use, add:

- survivorship-bias controls
- data quality monitoring
- realistic slippage/impact modeling
- portfolio and execution constraints
- reproducible experiment tracking

## Repository Hygiene

- Dependency list: `requirements.txt`
- Contribution guide: `CONTRIBUTING.md`
- License: `LICENSE`
