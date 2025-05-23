# Lesson 07 - 模型保存、加载与部署

本节课关注如何将训练好的模型迁移到生产环境，包括模型版本管理、API 服务化以及实时预测 Demo。

## 学习目标
1. 熟悉 Keras/TensorFlow 模型序列化与反序列化；
2. 了解常见的模型服务化方案（Flask/FastAPI、TensorFlow Serving、Docker 等）；
3. 能够快速搭建一个本地 RESTful API 用于波动率预测；
4. 理解部署中的性能监控与安全性考虑。

## 课堂内容
- 模型文件格式 (`.h5`, SavedModel) 对比
- 使用 FastAPI 编写推理接口
- Dockerfile 与容器化部署示例
- 前端可视化 (Streamlit) 简述

## 参考代码
本文件夹预留 `app/` 目录用于存放 API 代码及 Docker 配置，后续将在课堂中逐步补充。

## 课后作业
1. 完成 FastAPI 服务端点 `/predict`，输入 JSON 为最近 20 天特征序列，返回预测波动率；
2. 编写 `client_test.py` 进行接口压力测试；
3. （可选）将容器部署到 AWS/GCP 等公有云，并截屏佐证。 