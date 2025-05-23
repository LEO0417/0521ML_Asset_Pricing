#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hello_finance.py
----------------
示例脚本：从 Yahoo Finance 抓取资产价格并绘制收盘价。

用法：
    python hello_finance.py [TICKER] [START_DATE] [END_DATE]

参数说明：
    TICKER      股票或指数代码，默认为 ^GSPC（标普 500 指数）
    START_DATE  开始日期 (YYYY-MM-DD)，默认为上一自然年第一天
    END_DATE    结束日期 (YYYY-MM-DD)，默认为今天

该脚本将：
1. 下载数据（使用 yfinance）
2. 将数据保存为 CSV 文件
3. 绘制收盘价曲线并保存为 PNG
"""
import sys
from datetime import datetime, date
import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ 参数解析 ------------------
try:
    ticker = sys.argv[1]
except IndexError:
    ticker = "^GSPC"

this_year = date.today().year
try:
    default_start = date(this_year - 1, 1, 1).strftime("%Y-%m-%d")
except ValueError:
    default_start = "2023-01-01"

start_date = sys.argv[2] if len(sys.argv) > 2 else default_start
end_date = sys.argv[3] if len(sys.argv) > 3 else datetime.today().strftime("%Y-%m-%d")

print(f"Downloading {ticker} from {start_date} to {end_date} …")

# ------------------ 数据下载 ------------------
try:
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise ValueError("下载失败，返回数据为空，请检查代码或网络。")
except Exception as e:
    print(f"数据下载出现异常: {e}")
    sys.exit(1)

print(f"成功获取 {len(df)} 条记录。")

# ------------------ 保存 CSV ------------------
csv_path = f"{ticker.replace('^', '')}_prices.csv"
df.to_csv(csv_path)
print(f"数据已保存为 {csv_path}")

# ------------------ 绘图 ------------------
plt.figure(figsize=(10, 6))
plt.plot(df.index, df["Adj Close"], label="Adj Close", color="steelblue")
plt.title(f"{ticker} Price ({start_date} ~ {end_date})")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.legend()

img_path = f"{ticker.replace('^', '')}_price_plot.png"
plt.savefig(img_path, dpi=150)
plt.show()
print(f"图像已保存为 {img_path}") 