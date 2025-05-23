#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据获取与初步分析示例脚本
运行：python data_acquisition.py [TICKER]
"""
import sys
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

ticker = sys.argv[1] if len(sys.argv) > 1 else '^GSPC'
print(f'Downloading data for {ticker} …')

df = yf.download(ticker, start='2010-01-01', end='2023-12-31')
print(df.head())

# 描述性统计并保存
stats = df.describe()
stats.to_csv('descriptive_stats.csv')
print('Descriptive statistics saved to descriptive_stats.csv')

# 绘制收盘价
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Adj Close'])
plt.title(f'{ticker} Price')
plt.ylabel('Price')
plt.savefig('price_trend.png', dpi=150)
plt.show()

# 计算并绘制日收益率分布
df['Return'] = df['Adj Close'].pct_change()
plt.figure(figsize=(8, 4))
df['Return'].hist(bins=50)
plt.title('Daily Returns Distribution')
plt.savefig('return_hist.png', dpi=150)
plt.show()

# 保存数据
csv_path = f'{ticker.replace("^", "")}_prices.csv'
df.to_csv(csv_path)
print(f'Data saved to {csv_path}') 