# Lesson 02 - 开发环境配置与 Python 基础

本节课将指导同学完成本项目所需的开发环境配置，复习 Python 基础，并运行第一个示例脚本。

## 学习目标
1. 配置虚拟环境并安装依赖包；
2. 了解常用的金融数据包 (yfinance, pandas, numpy, matplotlib, seaborn 等)；
3. 复习 Python 数据结构与面向对象基础；
4. 运行 `hello_finance.py`，体验数据抓取与可视化。

## 环境配置步骤
```bash
# 1. 创建并激活虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Windows 使用 venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 验证安装
python -c "import pandas, numpy, matplotlib, yfinance, tensorflow, sklearn; print('环境就绪')"
```

## 示例脚本
在本文件夹下提供 `hello_finance.py`，代码片段如下：
```python
import yfinance as yf
import matplotlib.pyplot as plt

df = yf.download('^GSPC', start='2023-01-01', end='2023-12-31')
plt.plot(df.index, df['Adj Close'])
plt.title('S&P 500 2023 价格走势')
plt.show()
```

运行方法：
```bash
python hello_finance.py
```

## 课后作业
1. 在你的本地电脑完整跑通示例脚本并截图结果；
2. 选择一只股票（如 AAPL 或 TSLA），编写脚本下载并绘图；
3. 阅读 [TensorFlow 安装指南](https://www.tensorflow.org/install) 并确认你的 GPU 是否支持。 