import Functions_v2 as function
import numpy as np
import pandas as pd
import warnings
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore')
tqdm.pandas(desc="Processing Ticker")

# ----------------------------
# 配置参数
# ----------------------------
_ma_factor_cols = ['MaAreaSM', 'MaAreaML', 'MaBreakSM', 'MaBreakML', 'MaDurationSM', 'MaDurationML']
_mmt_factor_cols = ['mmt_normal_M', 'mmt_normal_A', 'mmt_avg_M', 'mmt_avg_A',
                    'mmt_intraday_M', 'mmt_intraday_A', 'mmt_overnight_M',
                    'mmt_overnight_A', 'mmt_range_M', 'mmt_range_A',
                    'mmt_route_M', 'mmt_route_A']
col_factor_full_list = _ma_factor_cols + _mmt_factor_cols

start_date_str = '2018-01-01'
end_date_str = '2024-05-31'

# Rolling window length (months)
ROLLING_MONTHS = 5

N_TOP_FACTORS = 3
FORWARD_RETURN_PERIOD = 21
VOLATILITY_LOOKBACK = 30
SL_THRESHOLD = 0.05
TP_THRESHOLD = 0.30
IC_WEIGHT_THRESHOLD_FOR_SIGNAL = 0.02

# Trigger settings
LONG_TRIGGER_Q = 0.8
SHORT_TRIGGER_Q = 0.2
LONG_TRIGGER_FALLBACK = 0.15
SHORT_TRIGGER_FALLBACK = -0.15


# ----------------------------
# 数据加载与预处理
# ----------------------------
print("加载数据...")
raw_df = pd.read_csv('combined_sp500_data.csv', parse_dates=['Date'])
raw_df['Date'] = pd.to_datetime(raw_df['Date'])

price_df_filtered_by_date = raw_df[
    (raw_df['Date'] >= pd.to_datetime(start_date_str)) &
    (raw_df['Date'] <= pd.to_datetime(end_date_str))
].copy()

if price_df_filtered_by_date.empty:
    raise ValueError(f"数据在日期范围 {start_date_str} 到 {end_date_str} 内为空。")

price_df_indexed = price_df_filtered_by_date.set_index(['Date', 'Ticker']).sort_index()

price_data_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
missing_price_cols_check = [col for col in price_data_cols if col not in price_df_indexed.columns]
if missing_price_cols_check:
    raise ValueError(f"价格数据中缺少列: {missing_price_cols_check}")

main_price_df = price_df_indexed[price_data_cols].copy()
main_price_df = main_price_df.dropna(subset=['Close'])

# Daily returns for the whole period (used for final performance calc)
def safe_daily_return_calculation(price_data_for_returns):
    if price_data_for_returns.empty:
        return pd.DataFrame()
    daily_returns = price_data_for_returns['Close'].groupby(level='Ticker').pct_change()
    return daily_returns.unstack(level='Ticker')

print("计算全样本日度收益率...")
all_periods_daily_returns_wide = safe_daily_return_calculation(main_price_df)
if all_periods_daily_returns_wide.empty:
    raise ValueError("日度收益率计算失败或为空。")

# Month-end dates available in your data
all_dates = main_price_df.index.get_level_values('Date')
month_ends = pd.DatetimeIndex(sorted(all_dates.unique())).to_period("M").to_timestamp("M").unique()
month_ends = pd.DatetimeIndex(month_ends)

# Only keep month-ends within [start, end]
start_dt = pd.to_datetime(start_date_str)
end_dt = pd.to_datetime(end_date_str)
month_ends = month_ends[(month_ends >= start_dt) & (month_ends <= end_dt)]

if len(month_ends) < (ROLLING_MONTHS + 1):
    raise ValueError("可用月末数量不足以做 rolling window，请检查数据范围。")


# ----------------------------
# 工具：逐日截面标准化（防泄露：只用当天截面）
# ----------------------------
def daily_cross_sectional_normalize(df_factors: pd.DataFrame) -> pd.DataFrame:
    """
    df_factors index: (Date, Ticker)
    Standardize cross-sectionally per Date: z = (x - mean_date) / std_date
    """
    if df_factors.empty:
        return df_factors

    out = []
    for date_val, grp in df_factors.groupby(level='Date'):
        mu = grp.mean(skipna=True)
        sd = grp.std(skipna=True).replace(0, np.nan)
        z = (grp - mu) / sd
        out.append(z)
    zdf = pd.concat(out).sort_index()
    return zdf.fillna(0)


# ----------------------------
# Rolling loop：每个 month-end T 估计参数 + 出信号
# ----------------------------
print(f"\n开始 rolling 回测：训练窗口={ROLLING_MONTHS}个月，TopN={N_TOP_FACTORS}，fwd={FORWARD_RETURN_PERIOD}d")
all_signals_list = []
debug_monthly_log = []  # 可选：记录每月 top factors / mean IC

# 从第 ROLLING_MONTHS-1 个月末开始才有完整训练窗口
# e.g., rolling 5 months: first usable T is the 5th month-end
for idx in tqdm(range(ROLLING_MONTHS - 1, len(month_ends) - 1), desc="Rolling Month-End"):
    T = month_ends[idx]

    # 训练窗口：过去 ROLLING_MONTHS 个月（含当月）
    # 取 T 往前推 ROLLING_MONTHS-1 个月的月初作为起点
    train_start_month = (T - pd.DateOffset(months=ROLLING_MONTHS - 1)).to_period("M").to_timestamp("M").replace(day=1)
    train_end = T

    # 下一月交易窗口（信号会在 calculate_performance 里 shift 到下月生效）
    next_month_end = month_ends[idx + 1]

    # ---- Slice price data for training window (<=T only; 防泄露关键) ----
    price_window = main_price_df.loc[
        (main_price_df.index.get_level_values('Date') >= train_start_month) &
        (main_price_df.index.get_level_values('Date') <= train_end)
    ].copy()

    if price_window.empty:
        continue

    # ---- Factor generation on window (uses only past to t in rolling stats) ----
    daily_factors_raw = function.generate_factors(price_window, s=15, m=30, l=120)
    if daily_factors_raw.empty:
        continue

    # Ensure factor list matches actual output (once)
    actual_cols = daily_factors_raw.columns.tolist()
    if set(actual_cols) != set(col_factor_full_list):
        col_factor_full_list = actual_cols

    # Winsorize / clip extreme values (no lookahead)
    daily_factors_cleaned = function.remove_extreme_value(daily_factors_raw)

    # Daily cross-sectional normalization (per date)
    daily_factors_normalized = daily_cross_sectional_normalize(daily_factors_cleaned)

    # ---- IC estimation (training window only) ----
    # NOTE: forward returns are computed within this window;
    # using future (t+21) is allowed only for training labels, not for trading.
    top_factors, mean_ic, monthly_ics = function.calculate_factor_ics_and_select_top(
        daily_normalized_factors=daily_factors_normalized,
        daily_prices_for_fwd_returns=price_window,
        n_top_factors=N_TOP_FACTORS,
        factor_cols_list=col_factor_full_list,
        forward_return_period=FORWARD_RETURN_PERIOD,
        ic_cutoff_date=T
    )

    if not top_factors:
        # fallback: take first N
        top_factors = col_factor_full_list[:N_TOP_FACTORS]
        if mean_ic is None or len(mean_ic) == 0:
            mean_ic = pd.Series(0.01, index=col_factor_full_list)

    # ---- Direction map: use sign(mean IC) for selected factors (更合理) ----
    factor_direction_map = pd.Series(1, index=col_factor_full_list, dtype=int)
    for f in top_factors:
        ic_val = mean_ic.get(f, 0.0)
        factor_direction_map.loc[f] = 1 if ic_val >= 0 else -1

    # ---- Monthly aggregation of normalized factors (window only) ----
    monthly_means = (
        daily_factors_normalized.swaplevel()
        .groupby(level='Ticker')
        .resample('M', level='Date')
        .mean()
        .dropna(how='all', axis=0)
    )
    if monthly_means.empty:
        continue

    # ---- Trigger thresholds from window monthly factors (NO future) ----
    long_trigger = monthly_means.quantile(LONG_TRIGGER_Q, axis=0)
    short_trigger = monthly_means.quantile(SHORT_TRIGGER_Q, axis=0)

    trigger_df = pd.DataFrame({
        'long_trigger': long_trigger,
        'short_trigger': short_trigger
    })
    # fill fallback
    trigger_df['long_trigger'] = trigger_df['long_trigger'].fillna(LONG_TRIGGER_FALLBACK)
    trigger_df['short_trigger'] = trigger_df['short_trigger'].fillna(SHORT_TRIGGER_FALLBACK)

    # ---- Build top_factors_names_df for a single month-end T ----
    top_factors_names_df = pd.DataFrame(
        [top_factors[:N_TOP_FACTORS]],
        index=pd.DatetimeIndex([T], name='Date'),
        columns=[f"top_{i+1}" for i in range(len(top_factors[:N_TOP_FACTORS]))]
    )

    # ---- factor_value_df: only (Ticker, T) rows (window T month-end) ----
    try:
        factor_value_T = monthly_means.xs(T, level='Date', drop_level=False)
    except KeyError:
        # if T not in monthly_means index (data missing), skip
        continue

    tickers_at_T = factor_value_T.index.get_level_values('Ticker').unique().tolist()
    if len(tickers_at_T) == 0:
        continue

    # available_tickers: for this rolling month-end, restrict to tickers we actually have at T
    available_tickers_map = {T.year: tickers_at_T}  # calculate_signals uses year lookup

    # ---- Generate signals for this month-end T (execute next month) ----
    signals_T = function.calculate_signals(
        factor_value_df=factor_value_T,                  # only (Ticker, T)
        top_factors_names_df=top_factors_names_df,        # only [T]
        trigger_df=trigger_df,
        available_tickers=available_tickers_map,
        factor_direction_map=factor_direction_map,
        factor_ic_map=mean_ic,
        ic_weight_threshold_for_signal=IC_WEIGHT_THRESHOLD_FOR_SIGNAL
    )

    if not signals_T.empty:
        all_signals_list.append(signals_T)

    # Optional debug log
    debug_monthly_log.append({
        "T": T,
        "train_start": train_start_month,
        "train_end": train_end,
        "trade_month_end": next_month_end,
        "top_factors": top_factors[:N_TOP_FACTORS],
        "mean_ic_top": {f: float(mean_ic.get(f, 0.0)) for f in top_factors[:N_TOP_FACTORS]}
    })


# ----------------------------
# 合并所有月末信号，并回测（calculate_performance 会 shift 到下月）
# ----------------------------
if not all_signals_list:
    raise ValueError("Rolling 过程中没有生成任何信号，请检查数据/阈值/IC门槛。")

rolling_signals_df = pd.concat(all_signals_list, axis=0).reset_index(drop=True)
rolling_signals_df['Date'] = pd.to_datetime(rolling_signals_df['Date'])
rolling_signals_df = rolling_signals_df.sort_values(['Date', 'Ticker'])

print("\nRolling 信号生成完毕，开始计算策略表现（信号下月生效）...")
nav_ts, perf_metrics, drawdown_series, daily_portfolio_returns = function.calculate_performance(
    signals_df=rolling_signals_df,
    daily_returns_wide_df=all_periods_daily_returns_wide,
    volatility_lookback=VOLATILITY_LOOKBACK,
    sl_threshold=SL_THRESHOLD,
    tp_threshold=TP_THRESHOLD,
    fully_invested=True,          # ✅ 建议：先开
    cooldown_after_exit=True,      # ✅ 建议：先开
    transaction_cost_bps=0.001,       # ✅ 10 bps
    cost_on_rebalance_only=True       # ✅ 如果想“每月调仓才收费”，就 True
    # cost_on_rebalance_only=False     # ✅ 如果想“止损/再归一等也会导致换手收费”，就 False
)

print("\n--- Rolling Backtest Performance ---")
if not nav_ts.empty:
    print(f"最终 NAV: {nav_ts.iloc[-1]:.4f}")
    print(f"总收益: {(nav_ts.iloc[-1] - 1):.2%}")
print(f"年化收益: {perf_metrics.get('Annualized_Return', np.nan):.3f}")
print(f"年化波动: {perf_metrics.get('Volatility', np.nan):.3f}")
print(f"夏普: {perf_metrics.get('Sharpe_Ratio', np.nan):.3f}")
print(f"最大回撤: {perf_metrics.get('Max_Drawdown', np.nan):.3f}")
print(f"历史VaR(95%, 日度): {perf_metrics.get('Historical_VaR_95', np.nan):.3f}")

# ----------------------------
# 可视化
# ----------------------------
print("\n生成图表...")
plt.figure(figsize=(14, 7))
if not nav_ts.empty:
    nav_ts.plot(label='Rolling Strategy NAV', lw=2)
plt.title(f'Rolling Strategy NAV (Window={ROLLING_MONTHS}m, TopN={N_TOP_FACTORS})')
plt.xlabel('Date')
plt.ylabel('NAV')
plt.legend()
plt.grid(True)
plt.savefig("rolling_strategy_nav.png")
plt.close()
print("NAV plot saved as rolling_strategy_nav.png")

plt.figure(figsize=(14, 7))
if not drawdown_series.empty:
    drawdown_series.plot(label='Drawdown', lw=1.5, alpha=0.8)
plt.title('Rolling Strategy Drawdown')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.legend()
plt.grid(True)
plt.savefig("rolling_strategy_drawdown.png")
plt.close()
print("Drawdown plot saved as rolling_strategy_drawdown.png")

# ---- Diagnostics ----
signals_cnt = rolling_signals_df['Action'].value_counts(dropna=False)
print("\n[Diagnostics] Signal counts:\n", signals_cnt)

# Rebuild daily actions the same way as calculate_performance (quick check)
sig = rolling_signals_df.copy()
sig['Date'] = pd.to_datetime(sig['Date'])
sig = sig.set_index(['Date','Ticker'])
actions_wide = sig['Action'].unstack('Ticker')
common = actions_wide.columns.intersection(all_periods_daily_returns_wide.columns)
actions_wide = actions_wide[common]
shifted = actions_wide.copy()
shifted.index = shifted.index + pd.offsets.MonthBegin(1)
daily_actions = shifted.reindex(all_periods_daily_returns_wide.index, method='ffill').fillna('hold')

active_cnt = daily_actions.isin(['long','short']).sum(axis=1)
print("\n[Diagnostics] Avg active positions per day:", active_cnt.mean())
print("[Diagnostics] Median active positions per day:", active_cnt.median())
print("[Diagnostics] % days with 0 position:", (active_cnt==0).mean())
print("[Diagnostics] % days with <=5 positions:", (active_cnt<=5).mean())

# market proxy = cross-sectional average of daily returns
mkt = all_periods_daily_returns_wide.mean(axis=1).fillna(0)

# beta = cov(strat, mkt)/var(mkt)
strat = daily_portfolio_returns.reindex(mkt.index).fillna(0)
beta = np.cov(strat, mkt)[0,1] / (np.var(mkt) + 1e-12)
corr = np.corrcoef(strat, mkt)[0,1]
print("\n[Diagnostics] beta vs equal-weight market proxy:", beta)
print("[Diagnostics] corr vs market proxy:", corr)

# ----------------------------
# SPX Benchmark Comparison
# ----------------------------
print("\nLoading SPX benchmark data for comparison...")
try:
    spx_raw = pd.read_csv("spx_data.csv", parse_dates=["Date"]).set_index("Date").sort_index()

    spx_price_col = "Adj Close" if "Adj Close" in spx_raw.columns else "Close"
    if spx_price_col not in spx_raw.columns:
        raise ValueError("spx_data.csv must contain 'Adj Close' or 'Close' column.")

    spx_px = spx_raw[spx_price_col].astype(float).rename("SPX_Price")
    spx_ret = spx_px.pct_change().fillna(0).rename("SPX_Return")

    # align dates to strategy daily returns index
    idx = daily_portfolio_returns.index
    spx_ret_aligned = spx_ret.reindex(idx).fillna(0)

    def perf_from_returns(r: pd.Series, ann_days: int = 252) -> dict:
        r = r.dropna()
        if r.empty:
            return {
                "Annualized_Return": np.nan,
                "Volatility": np.nan,
                "Sharpe_Ratio": np.nan,
                "Max_Drawdown": np.nan,
                "Historical_VaR_95": np.nan,
                "Final_NAV": np.nan
            }

        nav = (1 + r).cumprod()
        n = len(r)

        total_factor = nav.iloc[-1] / nav.iloc[0] if len(nav) > 1 else nav.iloc[-1]
        ann_ret = (total_factor ** (ann_days / n) - 1) if (total_factor > 0 and n > 0) else np.nan
        ann_vol = r.std(ddof=0) * np.sqrt(ann_days)

        sharpe = ann_ret / ann_vol if (ann_vol is not None and ann_vol > 1e-12) else np.nan

        peak = nav.expanding(min_periods=1).max()
        dd = (nav - peak) / peak.replace(0, np.nan)
        max_dd = abs(dd.min()) if not dd.empty else np.nan

        var95 = -np.percentile(r.values, 5) if len(r) >= 20 else np.nan

        return {
            "Annualized_Return": float(ann_ret) if pd.notna(ann_ret) else np.nan,
            "Volatility": float(ann_vol) if pd.notna(ann_vol) else np.nan,
            "Sharpe_Ratio": float(sharpe) if pd.notna(sharpe) else np.nan,
            "Max_Drawdown": float(max_dd) if pd.notna(max_dd) else np.nan,
            "Historical_VaR_95": float(var95) if pd.notna(var95) else np.nan,
            "Final_NAV": float(nav.iloc[-1]) if not nav.empty else np.nan,
        }, nav, dd

    # Strategy metrics already computed: perf_metrics + nav_ts + drawdown_series
    # We'll compute SPX metrics
    spx_metrics, spx_nav, spx_dd = perf_from_returns(spx_ret_aligned)

    # Build comparison table
    comparison_df = pd.DataFrame({
        "Strategy": {
            "Annualized Return": perf_metrics.get("Annualized_Return", np.nan),
            "Volatility": perf_metrics.get("Volatility", np.nan),
            "Sharpe Ratio": perf_metrics.get("Sharpe_Ratio", np.nan),
            "Max Drawdown": perf_metrics.get("Max_Drawdown", np.nan),
            "VaR 95% (Daily)": perf_metrics.get("Historical_VaR_95", np.nan),
            "Final NAV": nav_ts.iloc[-1] if not nav_ts.empty else np.nan,
        },
        "SPX": {
            "Annualized Return": spx_metrics["Annualized_Return"],
            "Volatility": spx_metrics["Volatility"],
            "Sharpe Ratio": spx_metrics["Sharpe_Ratio"],
            "Max Drawdown": spx_metrics["Max_Drawdown"],
            "VaR 95% (Daily)": spx_metrics["Historical_VaR_95"],
            "Final NAV": spx_metrics["Final_NAV"],
        }
    })

    def fmt(x):
        if pd.isna(x): return "NA"
        return f"{x:.3f}"

    print("\n--- Performance Comparison: Rolling Strategy vs SPX ---")
    print(comparison_df.applymap(fmt))

    # Plot normalized NAV comparison
    plt.figure(figsize=(14, 7))
    if not nav_ts.empty:
        (nav_ts / nav_ts.iloc[0]).plot(label="Strategy (Normalized NAV)", lw=2)
    if isinstance(spx_nav, pd.Series) and not spx_nav.empty:
        (spx_nav / spx_nav.iloc[0]).plot(label="SPX (Normalized NAV)", lw=2, linestyle="--")
    plt.title("Rolling Strategy vs SPX (Normalized NAV)")
    plt.xlabel("Date")
    plt.ylabel("Normalized NAV (Start=1)")
    plt.legend()
    plt.grid(True)
    plt.savefig("rolling_strategy_vs_spx.png")
    plt.close()
    print("Comparison plot saved as rolling_strategy_vs_spx.png")

except FileNotFoundError:
    print("Warning: spx_data.csv not found. Skipping SPX benchmark comparison.")
except Exception as e:
    print(f"Warning: Error occurred during SPX benchmark comparison: {e}")



print("\n脚本执行完毕。")
