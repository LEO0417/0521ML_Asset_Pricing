# Lesson 06 - 模型评估与可视化

本节课将对上节训练完成的模型进行深入评估、误差分析，并通过可视化手段展示模型效果。

## 学习目标
1. 理解 MSE、RMSE、MAE、R² 等常用回归指标；
2. 掌握使用 Matplotlib/Seaborn 绘制损失曲线与预测对比图；
3. 学会保存与加载 `.h5` Keras 模型文件；
4. 具备撰写技术报告中"实验结果"部分能力。

## 课堂内容
- 指标计算与意义解释
- 加载已有模型：`tf.keras.models.load_model()`
- 误差可视化与残差分析
- 结果图表在论文/幻灯片中的排版规范

## 文件说明
| 文件 | 说明 |
|------|------|
| `sp500_volatility_lstm_model.h5` | 上节保存的模型文件 |
| `assets/sp500_volatility_prediction_results.png` | 训练损失等多图布局 |
| `assets/sp500_volatility_prediction_scatter.png` | 预测值与实际值散点对比 |

## 运行示例
```python
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model = load_model('sp500_volatility_lstm_model.h5')
# 后续编写加载数据、预测并绘图的代码
```

## 课后作业
1. 实现 *残差直方图* 与 *QQ 图* 以检验误差分布；
2. 设计一个小实验，尝试使用 10 天窗口预测 10 天后的波动率，并比较指标差异；
3. 将所有结果汇总到一份 PDF 报告（指导模板见 `../lesson08_report`）。 