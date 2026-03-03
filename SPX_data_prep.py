import pandas as pd
import yfinance as yf
def get_sp500_tickers():
    """Fetches the list of S&P 500 tickers from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    df = table[0]
    return df['Symbol'].to_list()
def fetch_stock_data(tickers, start_date='2018-01-01', end_date='2024-12-31'):
    """Fetches historical stock data for a list of tickers."""
    data = {}
    for ticker in tickers:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        data[ticker] = stock_data
    return data
def save_to_csv(data, directory='sp500_data'):
    """Saves the fetched stock data to CSV files."""
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
    for ticker, df in data.items():
        df.to_csv(os.path.join(directory, f"{ticker}.csv"))
# Fetch S&P 500 tickers
sp500_tickers = get_sp500_tickers()
# Fetch historical stock data
stock_data = fetch_stock_data(sp500_tickers)
# Save data to CSV files
save_to_csv(stock_data)
print("S&P 500 stock data has been fetched and saved to CSV files.")

import os


def parse_custom_csv(file_path):
    """解析特殊格式的单个股票CSV文件"""
    # 从文件名获取股票代码
    ticker = os.path.basename(file_path).split('.')[0]

    # 读取原始数据，跳过无效行
    df = pd.read_csv(file_path, skiprows=3, header=None,
                     names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

    # 转换日期格式
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

    # 添加股票代码列
    df.insert(0, 'Ticker', ticker)

    # 重新排列列顺序
    return df[['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]


def concatenate_csv_files(directory='sp500_data', output_file='combined_sp500_data.csv'):
    """合并处理特殊格式的CSV文件"""
    all_data = []

    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            try:
                # 解析单个文件
                df = parse_custom_csv(file_path)
                all_data.append(df)
            except Exception as e:
                print(f"处理 {file} 失败: {str(e)}")
                continue

    # 合并数据
    combined = pd.concat(all_data, ignore_index=True)

    # 保存结果
    combined.to_csv(output_file, index=False)
    print(f"合并完成，共处理 {len(all_data)} 个文件")
    return combined


# 使用示例
final_data = concatenate_csv_files()
