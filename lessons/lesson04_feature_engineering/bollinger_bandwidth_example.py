#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""bollinger_bandwidth_example.py
计算布林带宽 (Bollinger Band Width) 并与原脚本整合演示。

• 读取 sp500_data_cache.csv（若不存在则调用 yfinance 下载标普500数据）
• 计算收盘价的 20 日布林带 (均线 ± 2 * std)
• 计算带宽： (Upper - Lower)/MA
• 绘制带宽随时间变化曲线
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

CACHE_FILE = os.path.join(os.path.dirname(__file__), '..', 'lesson05_model_building', 'sp500_data_cache.csv')


def load_data():
    if os.path.exists(CACHE_FILE):
        print("读取缓存数据 …")
        df = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        return df
    
    # 尝试下载数据
    try:
        print("下载数据 …")
        df = yf.download('^GSPC', start='2010-01-01', end='2023-12-31')
        if len(df) > 0:
            df.to_csv(CACHE_FILE)
            return df
    except:
        pass
    
    # 生成模拟数据
    print("网络连接失败，生成模拟数据 …")
    date_range = pd.date_range(start='2010-01-01', end='2023-12-31', freq='B')
    n_days = len(date_range)
    
    np.random.seed(42)
    initial_price = 1000.0
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
    
    return df


def compute_bb_width(df, window: int = 20):
    ma = df['Adj Close'].rolling(window).mean()
    std = df['Adj Close'].rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    bb_width = (upper - lower) / ma
    df[f'BBWidth_{window}'] = bb_width
    return df


def plot_bb_width(df, window: int = 20):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 上图：价格和布林带
    ax1.plot(df.index, df['Adj Close'], label='Price', color='blue', linewidth=1)
    
    ma = df['Adj Close'].rolling(window).mean()
    std = df['Adj Close'].rolling(window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    
    ax1.plot(df.index, ma, label=f'{window}-day MA', color='red', linewidth=1)
    ax1.fill_between(df.index, upper, lower, alpha=0.2, color='gray', label='Bollinger Bands')
    ax1.set_title(f'S&P500 Price with Bollinger Bands ({window}-day)')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 下图：布林带宽
    ax2.plot(df.index, df[f'BBWidth_{window}'], color='purple', linewidth=1)
    ax2.set_title(f'Bollinger Band Width ({window}-day)')
    ax2.set_ylabel('BB Width')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bollinger_band_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    df = load_data()
    print(f"数据加载成功，共 {len(df)} 条记录")
    
    df = compute_bb_width(df)
    
    print("布林带宽前5行数据：")
    bb_data = df[['Adj Close', 'BBWidth_20']].dropna()
    print(bb_data.head())
    
    print(f"\n布林带宽统计摘要：")
    print(f"平均带宽: {bb_data['BBWidth_20'].mean():.4f}")
    print(f"最小带宽: {bb_data['BBWidth_20'].min():.4f}")
    print(f"最大带宽: {bb_data['BBWidth_20'].max():.4f}")
    print(f"带宽标准差: {bb_data['BBWidth_20'].std():.4f}")
    
    plot_bb_width(df)
    
    # 保存结果
    bb_data.to_csv('bollinger_bandwidth_data.csv')
    print("\n结果已保存到 bollinger_bandwidth_data.csv")


if __name__ == '__main__':
    main() 