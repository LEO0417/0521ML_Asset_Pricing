#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
标普500指数波动性预测 - LSTM模型实践项目
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import time
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 1. 数据获取
def get_sp500_data(start_date='2010-01-01', end_date='2023-12-31', use_cache=True):
    """
    获取标普500指数历史数据，支持本地缓存以避免请求限制
    """
    cache_file = 'sp500_data_cache.csv'
    
    # 如果启用缓存且缓存文件存在，则直接读取缓存
    if use_cache and os.path.exists(cache_file):
        print(f"从本地缓存读取标普500指数数据...")
        sp500 = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"获取了 {len(sp500)} 天的标普500指数数据")
        return sp500
    
    print("正在从Yahoo Finance获取标普500指数数据...")
    try:
        # 添加延迟和重试机制，避免请求限制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                sp500 = yf.download('^GSPC', start=start_date, end=end_date)
                if len(sp500) > 0:
                    # 保存缓存
                    sp500.to_csv(cache_file)
                    print(f"获取了 {len(sp500)} 天的标普500指数数据")
                    print(f"数据已缓存到 {cache_file}")
                    return sp500
            except Exception as e:
                print(f"尝试 {attempt+1}/{max_retries} 失败: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 递增等待时间
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
        
        # 如果所有重试都失败，尝试使用示例数据
        print("无法从Yahoo Finance获取数据，使用示例数据...")
        # 生成示例数据
        date_rng = pd.date_range(start=start_date, end=end_date, freq='B')
        data = {
            'Open': np.random.normal(3000, 100, len(date_rng)),
            'High': np.random.normal(3050, 100, len(date_rng)),
            'Low': np.random.normal(2950, 100, len(date_rng)),
            'Close': np.random.normal(3000, 100, len(date_rng)),
            'Adj Close': np.random.normal(3000, 100, len(date_rng)),
            'Volume': np.random.normal(1000000, 200000, len(date_rng))
        }
        sp500 = pd.DataFrame(data, index=date_rng)
        print(f"生成了 {len(sp500)} 天的示例数据用于演示")
        return sp500
        
    except Exception as e:
        print(f"获取数据失败: {str(e)}")
        raise

# 2. 特征工程 - 计算历史波动率
def calculate_features(df, window=20):
    """
    计算特征，包括:
    - 历史收益率
    - 波动率(滚动标准差)
    - 交易量变化
    - 移动平均线
    - RSI指标
    """
    # 收益率计算
    df['return'] = df['Adj Close'].pct_change()
    
    # 波动率计算 (20天滚动窗口)
    df['volatility'] = df['return'].rolling(window=window).std() * np.sqrt(252)  # 年化
    
    # 简单移动平均
    df['SMA20'] = df['Adj Close'].rolling(window=20).mean()
    df['SMA50'] = df['Adj Close'].rolling(window=50).mean()
    
    # 成交量变化
    df['volume_change'] = df['Volume'].pct_change()
    
    # 价格变化率
    df['price_change_1d'] = df['Adj Close'].pct_change(1)
    df['price_change_5d'] = df['Adj Close'].pct_change(5)
    
    # 相对强弱指标(RSI)
    delta = df['Adj Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 波动率预测目标 - 未来5天的波动率
    df['future_volatility'] = df['volatility'].shift(-5)
    
    # 删除NaN值
    df.dropna(inplace=True)
    
    return df

# 3. 数据预处理
def prepare_data(df, target_col='future_volatility', seq_length=20, train_size=0.8):
    """
    准备LSTM模型的输入数据
    """
    # 选择特征列
    feature_cols = ['return', 'volatility', 'volume_change', 
                   'price_change_1d', 'price_change_5d', 'RSI']
    
    # 数据标准化
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # 标准化特征
    features_scaled = scaler_X.fit_transform(df[feature_cols])
    
    # 标准化目标
    target = df[target_col].values.reshape(-1, 1)
    target_scaled = scaler_y.fit_transform(target)
    
    # 创建序列数据
    X, y = [], []
    for i in range(len(features_scaled) - seq_length):
        X.append(features_scaled[i:i+seq_length])
        y.append(target_scaled[i+seq_length])
    
    X, y = np.array(X), np.array(y)
    
    # 划分训练集和测试集
    train_size = int(len(X) * train_size)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler_y

# 4. 构建LSTM模型
def build_lstm_model(seq_length, n_features):
    """
    构建LSTM模型
    """
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# 5. 模型训练与评估
def train_and_evaluate_model(X_train, y_train, X_test, y_test, scaler_y):
    """
    训练模型并评估性能
    """
    # 构建模型
    model = build_lstm_model(X_train.shape[1], X_train.shape[2])
    
    # 早停策略
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 预测
    y_pred_scaled = model.predict(X_test)
    
    # 反标准化
    y_test_actual = scaler_y.inverse_transform(y_test)
    y_pred_actual = scaler_y.inverse_transform(y_pred_scaled)
    
    # 计算评估指标
    mse = mean_squared_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)
    
    print(f"\n模型评估指标:")
    print(f"均方误差 (MSE): {mse:.6f}")
    print(f"均方根误差 (RMSE): {rmse:.6f}")
    print(f"平均绝对误差 (MAE): {mae:.6f}")
    print(f"决定系数 (R²): {r2:.6f}")
    
    return model, history, y_test_actual, y_pred_actual

# 6. 可视化结果
def visualize_results(df, history, y_test_actual, y_pred_actual):
    """
    可视化模型训练结果和预测结果
    """
    # 创建一个2行2列的图形布局
    fig = plt.figure(figsize=(18, 12))
    
    # 1. 绘制原始标普500指数走势
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(df.index[-500:], df['Adj Close'][-500:], label='标普500指数')
    ax1.set_title('标普500指数历史走势 (最近500个交易日)', fontsize=14)
    ax1.set_ylabel('价格', fontsize=12)
    ax1.grid(True)
    ax1.legend()
    
    # 2. 绘制历史波动率
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(df.index[-500:], df['volatility'][-500:], color='red', label='历史波动率')
    ax2.set_title('标普500指数历史波动率 (最近500个交易日)', fontsize=14)
    ax2.set_ylabel('波动率', fontsize=12)
    ax2.grid(True)
    ax2.legend()
    
    # 3. 绘制训练损失曲线
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(history.history['loss'], label='训练损失')
    ax3.plot(history.history['val_loss'], label='验证损失')
    ax3.set_title('模型训练与验证损失', fontsize=14)
    ax3.set_xlabel('迭代次数', fontsize=12)
    ax3.set_ylabel('损失', fontsize=12)
    ax3.grid(True)
    ax3.legend()
    
    # 4. 绘制预测vs实际波动率
    ax4 = fig.add_subplot(2, 2, 4)
    test_range = np.arange(len(y_test_actual))
    ax4.plot(test_range, y_test_actual, label='实际波动率')
    ax4.plot(test_range, y_pred_actual, label='预测波动率', linestyle='--')
    ax4.set_title('波动率预测对比', fontsize=14)
    ax4.set_xlabel('测试样本', fontsize=12)
    ax4.set_ylabel('波动率', fontsize=12)
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('sp500_volatility_prediction_results.png', dpi=300)
    plt.show()
    
    # 散点图对比实际值与预测值
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test_actual, y_pred_actual, alpha=0.5)
    plt.plot([y_test_actual.min(), y_test_actual.max()], 
             [y_test_actual.min(), y_test_actual.max()], 
             'r--', linewidth=2)
    plt.title('实际波动率 vs 预测波动率散点图', fontsize=14)
    plt.xlabel('实际波动率', fontsize=12)
    plt.ylabel('预测波动率', fontsize=12)
    plt.grid(True)
    plt.savefig('sp500_volatility_prediction_scatter.png', dpi=300)
    plt.show()

# 7. 主函数
def main():
    """
    主函数，执行整个波动率预测流程
    """
    print("=== 标普500指数波动率预测 - LSTM模型 ===")
    
    # 1. 获取数据
    df = get_sp500_data()
    
    # 2. 特征工程
    print("\n正在进行特征工程...")
    df = calculate_features(df)
    print(f"特征工程完成，特征包括: {df.columns.tolist()}")
    
    # 3. 数据准备
    print("\n正在准备模型输入数据...")
    X_train, X_test, y_train, y_test, scaler_y = prepare_data(df)
    print(f"训练数据形状: {X_train.shape}, {y_train.shape}")
    print(f"测试数据形状: {X_test.shape}, {y_test.shape}")
    
    # 4. 模型训练与评估
    print("\n开始训练LSTM模型...")
    model, history, y_test_actual, y_pred_actual = train_and_evaluate_model(
        X_train, y_train, X_test, y_test, scaler_y
    )
    
    # 5. 可视化结果
    print("\n正在生成可视化结果...")
    visualize_results(df, history, y_test_actual, y_pred_actual)
    
    # 6. 保存模型
    model.save('sp500_volatility_lstm_model.h5')
    print("\n模型已保存为 'sp500_volatility_lstm_model.h5'")
    
    # 7. 未来预测示例
    print("\n未来波动率预测示例:")
    last_sequence = X_test[-1:].copy()  # 获取最后一个序列作为预测起点
    
    for _ in range(5):  # 预测未来5天
        next_pred = model.predict(last_sequence)
        print(f"未来第{_+1}天预测波动率: {scaler_y.inverse_transform(next_pred)[0][0]:.6f}")
        
        # 更新序列以进行下一次预测
        # 假设简单移动最后一个序列，并添加预测值作为新的特征值
        # 注意：在实际应用中，这需要更复杂的处理来正确更新所有特征
        last_sequence = np.roll(last_sequence, -1, axis=1)
        
    print("\n=== 预测完成 ===")

if __name__ == "__main__":
    main() 