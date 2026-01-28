import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm


def remove_extreme_value(df_input):
    df_cleaned = df_input.copy()
    for column in df_cleaned.columns:
        if pd.api.types.is_numeric_dtype(df_cleaned[column]):
            if df_cleaned[column].count() < 2:
                continue
            Dm = df_cleaned[column].median()
            abs_diff_from_median = np.abs(df_cleaned[column] - Dm)
            if abs_diff_from_median.count() == 0:
                D1 = 0
            else:
                D1 = np.median(abs_diff_from_median.dropna())

            if D1 == 0:
                continue
            upper_bound = Dm + 5 * D1
            lower_bound = Dm - 5 * D1
            df_cleaned[column] = df_cleaned[column].clip(lower=lower_bound, upper=upper_bound)
    return df_cleaned


def standardize_data(df):
    df_copy = df.copy()
    group_level = None
    if isinstance(df_copy.index, pd.MultiIndex) and 'Date' in df_copy.index.names:
        group_level = 'Date'
    elif isinstance(df_copy.index, pd.DatetimeIndex):
        group_level = df_copy.index.name if df_copy.index.name is not None else 'Date'
        if df_copy.index.name is None:
            df_copy.index.name = 'Date'

    if group_level and ((isinstance(df_copy.index, pd.MultiIndex) and group_level in df_copy.index.names) or
                        (isinstance(df_copy.index, pd.DatetimeIndex))):
        centered_factors = df_copy.groupby(level=group_level).transform(lambda x: x - x.mean())
    else:
        centered_factors = df_copy.transform(lambda x: x - x.mean())

    overall_std = df_copy.std().replace(0, np.nan)
    standardized_factors = centered_factors / overall_std
    return standardized_factors


def generate_ma_std(df, s, m, l):
    df_copy = df.copy()
    df_copy[f'ma_{s}'] = df_copy['Close'].rolling(s, min_periods=1).mean().astype('float32')
    df_copy[f'ma_{m}'] = df_copy['Close'].rolling(m, min_periods=1).mean().astype('float32')
    df_copy[f'ma_{l}'] = df_copy['Close'].rolling(l, min_periods=1).mean().astype('float32')
    df_copy[f'std_{s}'] = df_copy['Close'].rolling(s, min_periods=1).std().astype('float32').fillna(0)
    df_copy[f'std_{m}'] = df_copy['Close'].rolling(m, min_periods=1).std().astype('float32').fillna(0)
    df_copy[f'std_{l}'] = df_copy['Close'].rolling(l, min_periods=1).std().astype('float32').fillna(0)
    return df_copy


def MA_Area(df_with_ma_std, s, m, l):
    df = df_with_ma_std.copy()
    ma_short = df[f'ma_{s}']
    ma_mid = df[f'ma_{m}']
    ma_long = df[f'ma_{l}']
    ma_diff_sm = ma_short - ma_mid
    ma_diff_ml = ma_mid - ma_long
    ma_sm_up_duration = (ma_short > ma_mid).astype(int)
    ma_sm_dn_duration = (ma_short <= ma_mid).astype(int)
    ma_ml_up_duration = (ma_mid > ma_long).astype(int)
    ma_ml_dn_duration = (ma_mid <= ma_long).astype(int)
    ma_sm_up_break = ((ma_diff_sm.shift(1) < 0) & (ma_diff_sm > 0)).astype(int)
    ma_sm_dn_break = ((ma_diff_sm.shift(1) > 0) & (ma_diff_sm < 0)).astype(int)
    ma_ml_up_break = ((ma_diff_ml.shift(1) < 0) & (ma_diff_ml > 0)).astype(int)
    ma_ml_dn_break = ((ma_diff_ml.shift(1) > 0) & (ma_diff_ml < 0)).astype(int)
    df['MaAreaSM'] = ma_diff_sm
    df['MaAreaML'] = ma_diff_ml
    df['MaBreakSM'] = ma_sm_up_break - ma_sm_dn_break
    df['MaBreakML'] = ma_ml_up_break - ma_ml_dn_break
    df['MaDurationSM'] = ma_sm_up_duration - ma_sm_dn_duration
    df['MaDurationML'] = ma_ml_up_duration - ma_ml_dn_duration
    return df[['MaAreaSM', 'MaAreaML', 'MaBreakSM', 'MaBreakML', 'MaDurationSM', 'MaDurationML']]


def momentum_reverse(df_with_ma_std, s, m, l):
    df = df_with_ma_std.copy()
    df['mmt_normal_M'] = df['Close'].pct_change(20)
    df['mmt_normal_A'] = df['Close'].pct_change(252) - df['mmt_normal_M']
    df['mmt_avg_M'] = df['Close'] / df['Close'].rolling(20, min_periods=1).mean() - 1
    rolling_mean_252 = df['Close'].rolling(252, min_periods=1).mean()
    df['mmt_avg_A'] = df['Close'].shift(20) / rolling_mean_252.replace(0, np.nan) - 1

    df['intraday_return_pct'] = (df['Close'] - df['Open']) / df['Open'].replace(0, np.nan)
    df['overnight_return_pct'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1).replace(0, np.nan)
    df['mmt_intraday_M'] = df['intraday_return_pct'].rolling(20, min_periods=1).sum()
    df['mmt_intraday_A'] = df['intraday_return_pct'].rolling(252, min_periods=1).sum()
    df['mmt_overnight_M'] = df['overnight_return_pct'].rolling(20, min_periods=1).sum()
    df['mmt_overnight_A'] = df['overnight_return_pct'].rolling(252, min_periods=1).sum()

    df['daily_return'] = df['Close'].pct_change()
    df['amplitude'] = (df['High'] - df['Low']) / df['Close'].shift(1).replace(0, np.nan)

    for window_suffix, window_days in [('_M', 20), ('_A', 252)]:
        df[f'threshold_top{window_suffix}'] = df['amplitude'].rolling(window=window_days, min_periods=1).quantile(0.8)
        df[f'threshold_bottom{window_suffix}'] = df['amplitude'].rolling(window=window_days, min_periods=1).quantile(0.2)
        df[f'top_amp{window_suffix}'] = df['amplitude'] >= df[f'threshold_top{window_suffix}']
        df[f'bottom_amp{window_suffix}'] = df['amplitude'] <= df[f'threshold_bottom{window_suffix}']
        df[f'top_amp_return{window_suffix}'] = df['daily_return'].where(df[f'top_amp{window_suffix}'])
        df[f'bottom_amp_return{window_suffix}'] = df['daily_return'].where(df[f'bottom_amp{window_suffix}'])
        df[f'top_amp_mean_ret{window_suffix}'] = df[f'top_amp_return{window_suffix}'].rolling(window=window_days, min_periods=1).mean()
        df[f'bottom_amp_mean_ret{window_suffix}'] = df[f'bottom_amp_return{window_suffix}'].rolling(window=window_days, min_periods=1).mean()
        df[f'mmt_range{window_suffix}'] = df[f'top_amp_mean_ret{window_suffix}'] - df[f'bottom_amp_mean_ret{window_suffix}']

    df['abs_daily_return_sum_M'] = df['daily_return'].abs().rolling(20, min_periods=1).sum()
    df['abs_daily_return_sum_A'] = df['daily_return'].abs().rolling(252, min_periods=1).sum()
    df['mmt_route_M'] = df['mmt_normal_M'] / df['abs_daily_return_sum_M'].replace(0, np.nan)
    df['mmt_route_A'] = (df['Close'].pct_change(252)) / df['abs_daily_return_sum_A'].replace(0, np.nan)

    expected_cols = ['mmt_normal_M', 'mmt_normal_A', 'mmt_avg_M', 'mmt_avg_A',
                     'mmt_intraday_M', 'mmt_intraday_A', 'mmt_overnight_M',
                     'mmt_overnight_A', 'mmt_range_M', 'mmt_range_A',
                     'mmt_route_M', 'mmt_route_A']
    return df[expected_cols]


func_list = [MA_Area, momentum_reverse]


def cal_factor(df_ticker_price_single_stock, s=15, m=30, l=120):
    df_with_ma_std = generate_ma_std(df_ticker_price_single_stock, s, m, l)
    all_factor_results_for_stock = []
    for factor_function in func_list:
        result_df = factor_function(df_with_ma_std, s, m, l)
        all_factor_results_for_stock.append(result_df)
    df_factor_all_types = pd.concat(all_factor_results_for_stock, axis=1)
    return df_factor_all_types


def generate_factors(price_data_multi_ticker, s=15, m=30, l=120):
    all_factors_list_all_stocks = []
    grouped_data = price_data_multi_ticker.groupby(level='Ticker')
    for ticker, ticker_price_data_group in grouped_data:
        single_ticker_date_indexed_df = ticker_price_data_group.droplevel('Ticker')
        try:
            factors_for_single_ticker = cal_factor(single_ticker_date_indexed_df, s, m, l)
            factors_for_single_ticker['Ticker'] = ticker
            all_factors_list_all_stocks.append(factors_for_single_ticker.reset_index())
        except Exception as e:
            print(f"Error processing ticker {ticker} in generate_factors: {str(e)}")
            continue
    if not all_factors_list_all_stocks:
        return pd.DataFrame(index=pd.MultiIndex(levels=[[], []], codes=[[], []], names=['Date', 'Ticker']))
    combined_factors_df_all_stocks = pd.concat(all_factors_list_all_stocks)
    return combined_factors_df_all_stocks.set_index(['Date', 'Ticker']).sort_index()


# ----------------------------
# ✅ 修改点 1：forward returns 增加 cutoff_date，强制切断未来
# ----------------------------
def calculate_forward_returns(price_data: pd.DataFrame, periods: int = 21, cutoff_date=None) -> pd.DataFrame:
    """
    price_data index: (Date, Ticker), must contain 'Close'
    Returns fwd_ret_{periods}d, but will set NaN if forward date > cutoff_date.
    This guarantees NO leakage beyond cutoff_date even if price_data includes future.
    """
    if price_data.empty or 'Close' not in price_data.columns:
        empty_idx = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['Date', 'Ticker'])
        return pd.DataFrame(index=empty_idx, columns=[f'fwd_ret_{periods}d'])

    df = price_data.reset_index().copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date'])

    # shift future close + future date within each ticker
    df[f'Close_fwd_{periods}d'] = df.groupby('Ticker')['Close'].shift(-periods)
    df[f'Date_fwd_{periods}d'] = df.groupby('Ticker')['Date'].shift(-periods)

    # hard cutoff: if forward date is after cutoff -> invalidate
    if cutoff_date is not None:
        cutoff_date = pd.to_datetime(cutoff_date)
        mask_bad = df[f'Date_fwd_{periods}d'].notna() & (df[f'Date_fwd_{periods}d'] > cutoff_date)
        df.loc[mask_bad, f'Close_fwd_{periods}d'] = np.nan

    fwd_ret_col_name = f'fwd_ret_{periods}d'
    df[fwd_ret_col_name] = (df[f'Close_fwd_{periods}d'] / df['Close']) - 1

    out = df[['Date', 'Ticker', fwd_ret_col_name]].set_index(['Date', 'Ticker']).sort_index()
    return out


# ----------------------------
# ✅ 修改点 2：IC计算函数增加 ic_cutoff_date，并传给 forward returns
# ----------------------------
def calculate_factor_ics_and_select_top(
    daily_normalized_factors: pd.DataFrame,
    daily_prices_for_fwd_returns: pd.DataFrame,
    n_top_factors,
    factor_cols_list,
    forward_return_period: int = 21,
    ic_cutoff_date=None
):
    print(f"   Calculating forward returns for IC (period: {forward_return_period} days)...")
    forward_returns = calculate_forward_returns(
        daily_prices_for_fwd_returns,
        periods=forward_return_period,
        cutoff_date=ic_cutoff_date
    )

    if forward_returns.empty or daily_normalized_factors.empty:
        print("Warning: Forward returns or daily factors are empty. Cannot calculate ICs.")
        empty_series = pd.Series(dtype=float, index=factor_cols_list)
        return [], empty_series, pd.DataFrame(index=pd.DatetimeIndex([]), columns=factor_cols_list)

    factors_for_ic = daily_normalized_factors[factor_cols_list].copy()
    data_for_ic = factors_for_ic.join(forward_returns, how='inner')
    data_for_ic = data_for_ic.dropna()

    if data_for_ic.empty:
        print("Warning: No aligned data for IC calculation after join/dropna.")
        empty_series = pd.Series(dtype=float, index=factor_cols_list)
        return [], empty_series, pd.DataFrame(index=pd.DatetimeIndex([]), columns=factor_cols_list)

    fwd_ret_col_name = f'fwd_ret_{forward_return_period}d'
    if fwd_ret_col_name not in data_for_ic.columns:
        print(f"Error: Forward return column '{fwd_ret_col_name}' not found in data_for_ic after join.")
        empty_series = pd.Series(dtype=float, index=factor_cols_list)
        return [], empty_series, pd.DataFrame(index=pd.DatetimeIndex([]), columns=factor_cols_list)

    print("   Calculating daily cross-sectional ICs...")
    daily_ics_list = []
    for date, group in tqdm(data_for_ic.groupby(level='Date'), desc="Daily IC", leave=False, mininterval=1.0):
        if group.shape[0] < 2:
            continue
        ics_for_date = {}
        for factor_col in factor_cols_list:
            if factor_col in group.columns:
                factor_values = group[factor_col]
                return_values = group[fwd_ret_col_name]
                valid_mask = factor_values.notna() & return_values.notna()
                if valid_mask.sum() > 1:
                    factor_values_clean = factor_values[valid_mask]
                    return_values_clean = return_values[valid_mask]
                    if factor_values_clean.std() > 1e-6 and return_values_clean.std() > 1e-6:
                        try:
                            corr, _ = pearsonr(factor_values_clean, return_values_clean)
                            ics_for_date[factor_col] = corr
                        except ValueError:
                            ics_for_date[factor_col] = np.nan
                    else:
                        ics_for_date[factor_col] = 0.0
                else:
                    ics_for_date[factor_col] = np.nan
            else:
                ics_for_date[factor_col] = np.nan
        if ics_for_date:
            daily_ics_list.append(pd.Series(ics_for_date, name=date))

    if not daily_ics_list:
        print("Warning: No daily ICs were calculated. Check data alignment or variance.")
        empty_series = pd.Series(dtype=float, index=factor_cols_list)
        return [], empty_series, pd.DataFrame(index=pd.DatetimeIndex([]), columns=factor_cols_list)

    all_daily_ics_df = pd.concat(daily_ics_list, axis=1).T
    all_daily_ics_df.index.name = 'Date'

    factor_mean_ic_series = all_daily_ics_df.mean().fillna(0)
    all_monthly_ics_df = all_daily_ics_df.resample('M').mean()

    print("   Selecting top N factors based on absolute mean IC...")
    top_n_factor_names_list = factor_mean_ic_series.abs().sort_values(ascending=False).head(n_top_factors).index.tolist()

    if not top_n_factor_names_list:
        print("Warning: No top factors selected by IC. Mean ICs might be all zero or NaN.")
    return top_n_factor_names_list, factor_mean_ic_series, all_monthly_ics_df


def calculate_signals(factor_value_df, top_factors_names_df, trigger_df,
                      available_tickers, factor_direction_map, factor_ic_map,
                      ic_weight_threshold_for_signal=0.0):
    signals = []
    if top_factors_names_df.empty or factor_value_df.empty:
        return pd.DataFrame(columns=['Date', 'Ticker', 'Action'])

    for date_month_end in tqdm(top_factors_names_df.index, desc="Generating signals", mininterval=1.0):
        year = date_month_end.year
        tickers_in_the_year = available_tickers.get(year, [])
        current_top_factor_names = top_factors_names_df.loc[date_month_end].dropna().values
        if not current_top_factor_names.size:
            continue

        for ticker in tickers_in_the_year:
            try:
                if not all(factor_name in factor_value_df.columns for factor_name in current_top_factor_names):
                    continue
                ticker_monthly_factor_values = factor_value_df.loc[(ticker, date_month_end), current_top_factor_names]

                buy_signal_contribution = 0
                sell_signal_contribution = 0

                for factor_name in current_top_factor_names:
                    if factor_name not in ticker_monthly_factor_values.index:
                        continue
                    factor_ic_weight = abs(factor_ic_map.get(factor_name, 0))
                    if factor_ic_weight < ic_weight_threshold_for_signal:
                        continue
                    direction = factor_direction_map.get(factor_name, 1)
                    current_factor_value = ticker_monthly_factor_values[factor_name]
                    if pd.isna(current_factor_value):
                        continue
                    if factor_name not in trigger_df.index:
                        continue
                    long_trigger = trigger_df.loc[factor_name, 'long_trigger']
                    short_trigger = trigger_df.loc[factor_name, 'short_trigger']

                    if direction == 1:
                        if current_factor_value > long_trigger:
                            buy_signal_contribution += factor_ic_weight
                        elif current_factor_value < short_trigger:
                            sell_signal_contribution += factor_ic_weight
                    elif direction == -1:
                        if current_factor_value < short_trigger:
                            buy_signal_contribution += factor_ic_weight
                        elif current_factor_value > long_trigger:
                            sell_signal_contribution += factor_ic_weight

                action = 'hold'
                if buy_signal_contribution > sell_signal_contribution and buy_signal_contribution > 0:
                    action = 'long'
                elif sell_signal_contribution > buy_signal_contribution and sell_signal_contribution > 0:
                    action = 'short'

                signals.append({'Date': date_month_end, 'Ticker': ticker, 'Action': action})
            except KeyError:
                continue
            except Exception:
                continue

    if not signals:
        return pd.DataFrame(columns=['Date', 'Ticker', 'Action'])

    signals_df = pd.DataFrame(signals)
    if not signals_df.empty:
        signals_df['Date'] = pd.to_datetime(signals_df['Date'])
    return signals_df


def calculate_performance(
    signals_df, daily_returns_wide_df,
    volatility_lookback=60,
    sl_threshold=None, tp_threshold=None,
    fully_invested=True,
    cooldown_after_exit=True,
    transaction_cost_bps=0.001,     # ✅ 10 bps = 0.001（按 turnover）
    cost_on_rebalance_only=False    # ✅ True=只在月初调仓日收费；False=每天按实际 turnover 收费
):
    if signals_df.empty:
        empty_cum_ret = pd.Series(dtype=float, name="Cumulative_Return_NAV")
        empty_drawdown = pd.Series(dtype=float, name="Drawdown")
        empty_perf_metrics = {'Annualized_Return': np.nan, 'Volatility': np.nan, 'Sharpe_Ratio': np.nan,
                              'Max_Drawdown': np.nan, 'Historical_VaR_95': np.nan}
        return empty_cum_ret, empty_perf_metrics, empty_drawdown, pd.Series(dtype=float)

    signals_df_processed = signals_df.copy()
    signals_df_processed['Date'] = pd.to_datetime(signals_df_processed['Date'])
    if not (isinstance(signals_df_processed.index, pd.MultiIndex) and
            signals_df_processed.index.names == ['Date', 'Ticker']):
        signals_df_processed = signals_df_processed.set_index(['Date', 'Ticker'])

    actions_wide = signals_df_processed['Action'].unstack(level='Ticker')
    common_tickers = actions_wide.columns.intersection(daily_returns_wide_df.columns)

    if common_tickers.empty:
        empty_idx = daily_returns_wide_df.index[:1]
        empty_cum_ret = pd.Series(dtype=float, name="Cumulative_Return_NAV", index=empty_idx)
        empty_drawdown = pd.Series(dtype=float, name="Drawdown", index=empty_idx)
        empty_perf_metrics = {'Annualized_Return': np.nan, 'Volatility': np.nan, 'Sharpe_Ratio': np.nan,
                              'Max_Drawdown': np.nan, 'Historical_VaR_95': np.nan}
        return empty_cum_ret, empty_perf_metrics, empty_drawdown, pd.Series(dtype=float, index=empty_idx)

    actions_wide_common = actions_wide[common_tickers]
    returns_df_common = daily_returns_wide_df[common_tickers].fillna(0)

    # 月末信号 -> 下月生效
    shifted_actions = actions_wide_common.copy()
    shifted_actions.index = shifted_actions.index + pd.offsets.MonthBegin(1)
    daily_actions_original = shifted_actions.reindex(returns_df_common.index, method='ffill').fillna('hold')

    # --- SL/TP + cooldown（当月止损后禁止再入场） ---
    daily_actions = daily_actions_original.copy()
    if sl_threshold is not None or tp_threshold is not None:
        print(f"   Applying SL: {sl_threshold}, TP: {tp_threshold} logic...")

        entry_nav = pd.Series(1.0, index=common_tickers)
        pos_nav = pd.Series(1.0, index=common_tickers)
        pos_state = pd.Series('hold', index=common_tickers)  # 'long','short','hold'
        exited_this_month = pd.Series(False, index=common_tickers)

        prev_month = None
        for date in daily_actions.index:
            cur_month = (date.year, date.month)
            if prev_month is None or cur_month != prev_month:
                exited_this_month[:] = False
                prev_month = cur_month

            for tkr in common_tickers:
                orig = daily_actions_original.loc[date, tkr]
                r = returns_df_common.loc[date, tkr]

                if cooldown_after_exit and exited_this_month.loc[tkr]:
                    daily_actions.loc[date, tkr] = 'hold'
                    pos_state.loc[tkr] = 'hold'
                    continue

                if pos_state.loc[tkr] == 'hold':
                    if orig in ['long', 'short']:
                        pos_state.loc[tkr] = orig
                        entry_nav.loc[tkr] = 1.0
                        pos_nav.loc[tkr] = 1.0 * (1 + (r if orig == 'long' else -r))
                        daily_actions.loc[date, tkr] = orig
                    else:
                        daily_actions.loc[date, tkr] = 'hold'
                elif pos_state.loc[tkr] == 'long':
                    pos_nav.loc[tkr] *= (1 + r)
                    pnl = pos_nav.loc[tkr] / entry_nav.loc[tkr] - 1
                    hit_sl = (sl_threshold is not None and pnl < -sl_threshold)
                    hit_tp = (tp_threshold is not None and pnl > tp_threshold)
                    if hit_sl or hit_tp:
                        daily_actions.loc[date, tkr] = 'hold'
                        pos_state.loc[tkr] = 'hold'
                        if cooldown_after_exit:
                            exited_this_month.loc[tkr] = True
                    else:
                        daily_actions.loc[date, tkr] = 'long'
                elif pos_state.loc[tkr] == 'short':
                    pos_nav.loc[tkr] *= (1 - r)
                    pnl = pos_nav.loc[tkr] / entry_nav.loc[tkr] - 1
                    hit_sl = (sl_threshold is not None and pnl < -sl_threshold)
                    hit_tp = (tp_threshold is not None and pnl > tp_threshold)
                    if hit_sl or hit_tp:
                        daily_actions.loc[date, tkr] = 'hold'
                        pos_state.loc[tkr] = 'hold'
                        if cooldown_after_exit:
                            exited_this_month.loc[tkr] = True
                    else:
                        daily_actions.loc[date, tkr] = 'short'

    # --- 每只股票策略日收益（long=+r, short=-r） ---
    strat_ret = pd.DataFrame(0.0, index=daily_actions.index, columns=common_tickers)
    long_mask = (daily_actions == 'long')
    short_mask = (daily_actions == 'short')
    strat_ret[long_mask] = returns_df_common[long_mask]
    strat_ret[short_mask] = -returns_df_common[short_mask]

    # --- Risk parity weighting + ✅交易成本（turnover）---
    print("   Applying Risk Parity weighting...")
    daily_portfolio_return = pd.Series(0.0, index=daily_actions.index)
    daily_cost = pd.Series(0.0, index=daily_actions.index)

    weights = pd.Series(dtype=float)  # 月初算出来的 base weights（只在 rebalance 更新）
    prev_signed_alloc = pd.Series(0.0, index=common_tickers)  # 用于 turnover

    for i, date in enumerate(tqdm(daily_actions.index, desc="Risk Parity & NAV", mininterval=1.0, leave=False)):
        rebalance = (i == 0) or (date.month != daily_actions.index[i-1].month)

        # 当天 active
        active = daily_actions.loc[date][daily_actions.loc[date].isin(['long','short'])].index

        # 月初更新 base weights
        if rebalance:
            if len(active) > 0:
                if i > volatility_lookback:
                    hist = returns_df_common.loc[:daily_actions.index[i-1], active]
                    vol = hist.rolling(window=volatility_lookback,
                                       min_periods=max(1, volatility_lookback//2)).std().iloc[-1]
                    vol = vol.replace(0, np.nan).fillna(method='bfill').fillna(method='ffill').fillna(0.0001)
                    inv = 1 / vol
                    weights = inv / inv.sum()
                else:
                    weights = pd.Series(1/len(active), index=active)
            else:
                weights = pd.Series(dtype=float)

        # 当天用于收益计算的权重（fully-invested 则对 active 再归一）
        if weights.empty or len(active) == 0:
            signed_alloc_today = pd.Series(0.0, index=common_tickers)
            gross = 0.0
        else:
            if fully_invested:
                w = weights.reindex(active).dropna()
                if len(w) > 0 and w.sum() > 0:
                    w = w / w.sum()
                else:
                    w = pd.Series(dtype=float)
            else:
                # 允许现金：inactive 权重视为 0，不再归一
                w = weights.reindex(active).dropna()

            if w.empty:
                signed_alloc_today = pd.Series(0.0, index=common_tickers)
                gross = 0.0
            else:
                # gross return
                gross = float((w * strat_ret.loc[date, w.index]).sum())

                # signed allocation for turnover
                sign = daily_actions.loc[date, w.index].map({'long': 1.0, 'short': -1.0}).astype(float)
                signed_part = (w * sign)

                signed_alloc_today = pd.Series(0.0, index=common_tickers)
                signed_alloc_today.loc[signed_part.index] = signed_part.values

        # ✅ turnover cost
        if transaction_cost_bps is not None and transaction_cost_bps > 0:
            if (not cost_on_rebalance_only) or rebalance:
                turnover = 0.5 * float((signed_alloc_today - prev_signed_alloc).abs().sum())
                cost = transaction_cost_bps * turnover
            else:
                cost = 0.0
        else:
            cost = 0.0

        daily_cost.loc[date] = cost
        daily_portfolio_return.loc[date] = gross - cost

        prev_signed_alloc = signed_alloc_today  # 更新用于下一天 turnover

    daily_portfolio_return = daily_portfolio_return.fillna(0)

    nav = (1 + daily_portfolio_return).cumprod()
    if not nav.empty and pd.isna(nav.iloc[0]):
        nav.iloc[0] = 1.0

    # --- metrics ---
    days_in_year = 252
    n = len(daily_portfolio_return)

    if n > 0:
        total_factor = nav.iloc[-1] / nav.iloc[0]
        annual_return = total_factor ** (days_in_year / n) - 1
        annual_vol = daily_portfolio_return.std() * np.sqrt(days_in_year)
        sharpe = annual_return / annual_vol if annual_vol > 1e-12 else np.nan
    else:
        annual_return = np.nan
        annual_vol = np.nan
        sharpe = np.nan

    peak = nav.expanding(min_periods=1).max()
    drawdown = (nav - peak) / peak.replace(0, np.nan)
    mdd = abs(drawdown.min()) if not drawdown.empty else np.nan

    var95 = -np.percentile(daily_portfolio_return.dropna(), 5) if n > 20 else np.nan

    perf = {
        'Annualized_Return': annual_return,
        'Volatility': annual_vol,
        'Sharpe_Ratio': sharpe,
        'Max_Drawdown': mdd,
        'Historical_VaR_95': var95,
        'Avg_Daily_Cost': float(daily_cost.mean()),
        'Total_Cost': float(daily_cost.sum()),
    }
    return nav, perf, drawdown.fillna(0), daily_portfolio_return


