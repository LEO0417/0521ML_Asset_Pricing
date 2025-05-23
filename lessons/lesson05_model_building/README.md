# Lesson 05 - LSTM 模型构建与训练

本节课进入深度学习模型的核心：使用 LSTM 网络预测标普 500 指数波动率。

## 学习目标
1. 理解 RNN 与 LSTM 的基本原理与门控机制；
2. 熟悉 Keras 构建多层 LSTM 的常用 API；
3. 掌握 EarlyStopping、Dropout 等技巧以防止过拟合；
4. 能够读懂并修改 `sp500_volatility_prediction.py`。

## 课堂内容
- 循环神经网络 (RNN) 与 LSTM 结构剖析
- 代码讲解：`build_lstm_model()`、`train_and_evaluate_model()`
- 模型超参数调优：层数、隐藏单元、学习率、批量大小

## 运行示例
```bash
cd lessons/lesson05_model_building
python sp500_volatility_prediction.py  # 训练并自动保存模型
```

训练完成后将得到：
- `../lesson06_evaluation_and_visualization/sp500_volatility_lstm_model.h5`
- 日志与指标输出

## 课后作业
1. 通过修改隐藏层数量和单元数，再训练两种新模型，并记录评估指标；
2. 确认运行结束后，观察 GPU/CPU 占用与训练耗时；
3. 将最佳模型保存为 `best_lstm_model.h5` 并说明为何是最佳。 