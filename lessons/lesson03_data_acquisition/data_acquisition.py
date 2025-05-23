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
import numpy as np
import os

sns.set_style('darkgrid')

ticker = sys.argv[1] if len(sys.argv) > 1 else '^GSPC'
print(f'Downloading data for {ticker} …')

# 尝试从缓存读取
cache_file = f'../lesson05_model_building/sp500_data_cache.csv'
if ticker == '^GSPC' and os.path.exists(cache_file):
    print("找到缓存文件，直接使用...")
    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
else:
    # 尝试下载数据
    try:
        df = yf.download(ticker, start='2010-01-01', end='2023-12-31')
        if df.empty:
            raise ValueError("数据为空")
    except:
        print("网络获取失败，生成模拟数据用于演示...")
        # 生成模拟数据
        date_range = pd.date_range(start='2010-01-01', end='2023-12-31', freq='B')
        n_days = len(date_range)
        
        np.random.seed(42)
        initial_price = 1000.0 if ticker == '^GSPC' else 100.0
        returns = np.random.normal(0.0005, 0.015, n_days)
        prices = [initial_price]
        
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))
        
        df = pd.DataFrame({
            'Open': np.array(prices) * (1 + np.random.normal(0, 0.001, n_days)),
            'High': np.array(prices) * (1 + np.abs(np.random.normal(0, 0.003, n_days))),
            'Low': np.array(prices) * (1 - np.abs(np.random.normal(0, 0.003, n_days))),
            'Close': prices,
            'Adj Close': prices,
            'Volume': np.random.randint(1000000, 10000000, n_days)
        }, index=date_range)

print(f"数据获取成功，共 {len(df)} 条记录")
print("前5行数据：")
print(df.head())

# 描述性统计并保存
stats = df.describe()
stats.to_csv('descriptive_stats.csv')
print('Descriptive statistics saved to descriptive_stats.csv')

# 绘制收盘价
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(df.index, df['Adj Close'])
plt.title(f'{ticker} Price Trend')
plt.ylabel('Price')
plt.grid(True)

# 计算并绘制日收益率分布
df['Return'] = df['Adj Close'].pct_change()
plt.subplot(2, 2, 2)
df['Return'].hist(bins=50)
plt.title('Daily Returns Distribution')
plt.xlabel('Return')
plt.ylabel('Frequency')

# 绘制成交量
plt.subplot(2, 2, 3)
plt.plot(df.index, df['Volume'])
plt.title('Trading Volume')
plt.ylabel('Volume')
plt.grid(True)

# 绘制价格箱线图（按年）
plt.subplot(2, 2, 4)
df['Year'] = df.index.year
yearly_returns = [df[df['Year'] == year]['Return'].dropna() for year in df['Year'].unique()[-5:]]
if yearly_returns:
    plt.boxplot(yearly_returns, labels=df['Year'].unique()[-5:])
    plt.title('Returns by Year (Last 5 Years)')
    plt.ylabel('Return')

plt.tight_layout()
plt.savefig('data_analysis_summary.png', dpi=150)
plt.show()

# 保存数据
csv_path = f'{ticker.replace("^", "")}_prices.csv'
df.to_csv(csv_path)
print(f'Data saved to {csv_path}')

# 输出关键统计信息
print(f'\n=== {ticker} 数据分析摘要 ===')
print(f'数据时间范围: {df.index.min()} 到 {df.index.max()}')
print(f'价格范围: {df["Adj Close"].min():.2f} - {df["Adj Close"].max():.2f}')
print(f'平均日收益率: {df["Return"].mean():.4f} ({df["Return"].mean()*252:.2f}% 年化)')
print(f'收益率波动率: {df["Return"].std():.4f} ({df["Return"].std()*np.sqrt(252):.2f}% 年化)')
print(f'最大回撤: {((df["Adj Close"] / df["Adj Close"].cummax()) - 1).min():.2%}') 