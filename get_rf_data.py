import pandas_datareader.data as web
import pandas as pd
import datetime


def download_rf_from_fred(start_date='2015-01-01', output_file='risk_free_rate.csv'):
    print(f"尝试从 FRED 下载 3个月国债利率 (DTB3)...")

    start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.datetime.now()

    try:
        # DTB3 是 3-Month Treasury Bill: Secondary Market Rate
        rf_df = web.DataReader('DTB3', 'fred', start, end)

        # FRED 数据也是百分比 (e.g., 4.25)，需要转小数再转日度
        # 填充 NaN (周末/节假日)
        rf_df = rf_df.fillna(method='ffill')

        # 计算日度无风险利率
        # 1. 除以 100 转小数
        # 2. 除以 252 转日度
        rf_daily = (rf_df['DTB3'] / 100) / 252

        rf_daily.name = 'rf_daily'
        rf_daily.index.name = 'Date'

        rf_daily.to_csv(output_file, header=True)
        print(f"成功! 数据已保存至 {output_file}")
        print(rf_daily.tail())

    except Exception as e:
        print(f"FRED 下载失败: {e}")


if __name__ == "__main__":
    download_rf_from_fred()