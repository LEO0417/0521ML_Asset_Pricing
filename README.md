# 标普500指数波动率预测 - LSTM模型项目

本项目是"机器学习与资产定价"课程的实践案例，使用LSTM深度学习模型预测标普500指数的未来波动率。

## 项目概述

- **研究对象**：标普500指数（S&P 500）
- **研究目标**：预测未来短期（5个交易日）波动率
- **模型选择**：长短期记忆网络（LSTM）
- **数据范围**：2010-2023年每日交易数据

## 项目特点

1. **完整的机器学习流程**：从数据获取、预处理、特征工程到模型构建、训练与评估
2. **时间序列预测**：处理金融时间序列数据的特殊性
3. **深度学习应用**：应用LSTM网络捕捉金融数据中的长期依赖关系
4. **实用性强**：可直接应用于实际金融市场分析

## 环境配置

```bash
# 安装所需依赖
pip install -r requirements.txt
```

## 使用方法

```bash
# 运行主程序
python sp500_volatility_prediction.py
```

## 项目输出

1. 训练好的LSTM模型（保存为.h5文件）
2. 模型评估指标（MSE、RMSE、MAE、R²）
3. 可视化结果图表（指数走势、历史波动率、预测对比等）
4. 未来5天波动率预测值

## 教学意义

1. 掌握金融数据处理技术
2. 理解机器学习在金融领域的应用
3. 熟悉深度学习模型构建和评估
4. 培养数据分析和可视化能力

## 参考文献

- Heaton, J. B., Polson, N. G., & Witte, J. H. (2017). Deep learning for finance: deep portfolios. Applied Stochastic Models in Business and Industry, 33(1), 3-12.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. The Review of Financial Studies, 33(5), 2223-2273.

# 机器学习与资产定价短期课程教案

本存储库包含 **机器学习与资产定价** 短期课程的全部教学资源。为了便于按节次使用，本仓库将所有内容按照课程进度拆分到 `lessons/` 目录下的子文件夹中，每一个子文件夹代表一节课。

## 目录结构

```
lessons/
├── lesson01_overview                  # 课程介绍与学习目标
├── lesson02_environment               # 开发环境配置与 Python 基础
├── lesson03_data_acquisition          # 数据获取与初步分析
├── lesson04_feature_engineering       # 特征工程与数据预处理
├── lesson05_model_building            # LSTM 模型构建与训练
├── lesson06_evaluation_and_visualization # 模型评估与可视化
├── lesson07_deployment                # 模型保存、加载与部署
└── lesson08_report                    # 项目报告与课程总结
```

## 快速开始

1. 克隆仓库并进入
   ```bash
   git clone <repo_url>
   cd <repo_name>
   ```
2. 创建并激活虚拟环境（可选）
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows 请使用 venv\Scripts\activate
   ```
3. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```
4. 进入对应节次文件夹，根据 `README.md` 步骤运行示例代码。

---

> 如对课程资料有任何疑问，欢迎在 Issues 中反馈。 