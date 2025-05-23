#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FastAPI 服务：波动率预测 API

运行方式：
    uvicorn main:app --reload

请求示例：
POST /predict
{
  "sequence": [[0.1, 0.2, ...], [...], ...]   # shape = [seq_length, n_features]
}
返回：
{
  "predicted_volatility": 0.12345
}
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import numpy as np
import tensorflow as tf
import os

# ---------- 配置 ----------
SEQ_LENGTH = 20
N_FEATURES = 6  # 与训练时保持一致
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'sp500_volatility_lstm_model.h5')

# ---------- 加载模型 ----------
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"模型加载失败，请确认路径是否正确: {e}")

# ---------- 定义输入输出 Schema ----------
class SequenceIn(BaseModel):
    sequence: conlist(conlist(float, min_items=N_FEATURES, max_items=N_FEATURES), min_items=SEQ_LENGTH, max_items=SEQ_LENGTH)

class PredictionOut(BaseModel):
    predicted_volatility: float

# ---------- 启动应用 ----------
app = FastAPI(title="S&P500 Volatility Prediction API", version="0.1.0")

@app.get("/")
def root():
    return {"message": "Welcome to Volatility Prediction API"}

@app.post("/predict", response_model=PredictionOut)
def predict(input_data: SequenceIn):
    try:
        # 预处理: 转为 numpy 数组，扩展 batch 维度
        seq_array = np.array(input_data.sequence, dtype=np.float32).reshape(1, SEQ_LENGTH, N_FEATURES)
        pred_scaled = model.predict(seq_array)
        # 模型输出已反归一化? 若否，可在此加载 scaler 进行反变换
        return {"predicted_volatility": float(pred_scaled[0][0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) 