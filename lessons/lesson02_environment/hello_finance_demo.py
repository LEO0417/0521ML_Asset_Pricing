#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
hello_finance_demo.py
--------------------
示例脚本（演示版）：生成模拟数据并绘制，避免网络限制。

用法：
    python hello_finance_demo.py [TICKER] [START_DATE] [END_DATE]
"""
import sys
from datetime import datetime, date
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 参数解析
ticker = sys.argv[1] if len(sys.argv) > 1 else "DEMO_STOCK"
start_date = sys.argv[2] if len(sys.argv) > 2 else "2023-01-01"
end_date = sys.argv[3] if len(sys.argv) > 3 else "2023-12-31"

print(f"生成模拟数据: {ticker} from {start_date} to {end_date}")

# 生成模拟数据
date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 工作日
n_days = len(date_range)

# 生成随机游走价格数据
np.random.seed(42)  # 确保可重复
initial_price = 100.0
returns = np.random.normal(0.001, 0.02, n_days)  # 日收益率
prices = [initial_price]

for r in returns[1:]:
    prices.append(prices[-1] * (1 + r))

# 创建DataFrame
df = pd.DataFrame({
    'Open': np.array(prices) * (1 + np.random.normal(0, 0.001, n_days)),
    'High': np.array(prices) * (1 + np.abs(np.random.normal(0, 0.005, n_days))),
    'Low': np.array(prices) * (1 - np.abs(np.random.normal(0, 0.005, n_days))),
    'Close': prices,
    'Adj Close': prices,
    'Volume': np.random.randint(1000000, 5000000, n_days)
}, index=date_range)

print(f"生成了 {len(df)} 条记录")
print("前5行数据：")
print(df.head())

# 保存CSV
csv_path = f"{ticker}_demo_prices.csv"
df.to_csv(csv_path)
print(f"数据已保存为 {csv_path}")

# 绘图
plt.figure(figsize=(12, 8))

# 子图1：价格走势
plt.subplot(2, 1, 1)
plt.plot(df.index, df['Adj Close'], label="Adj Close", color="steelblue", linewidth=1.5)
plt.title(f"{ticker} Price Trend ({start_date} ~ {end_date})")
plt.ylabel("Price")
plt.grid(True, alpha=0.3)
plt.legend()

# 子图2：成交量
plt.subplot(2, 1, 2)
plt.bar(df.index, df['Volume'], color='orange', alpha=0.7, width=1)
plt.title("Trading Volume")
plt.ylabel("Volume")
plt.grid(True, alpha=0.3)

plt.tight_layout()
img_path = f"{ticker}_demo_analysis.png"
plt.savefig(img_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"图像已保存为 {img_path}")

# 简单统计分析
df['Return'] = df['Adj Close'].pct_change()
print(f"\n{ticker} 统计摘要:")
print(f"价格范围: ${df['Adj Close'].min():.2f} - ${df['Adj Close'].max():.2f}")
print(f"平均日收益率: {df['Return'].mean():.4f}")
print(f"收益率标准差: {df['Return'].std():.4f}")
print(f"总收益率: {(df['Adj Close'].iloc[-1]/df['Adj Close'].iloc[0]-1)*100:.2f}%") 