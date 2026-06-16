# EpiAI 概述

EpiAI 是一个端到端的传染病爆发预测框架，支持深度学习、机器学习
和传统时间序列模型。提供统一的数据管道、模型注册系统、训练入口、
多模型管理（ModelVault）和生产部署运行时（DeploymentRuntime）。

---

## 核心能力

| 能力 | 说明 |
|------|------|
| **统一数据管道** | CSV / Feather → 拆分 → 变换 → 滑窗，一行代码完成 |
| **三大模型族** | Torch / Sklearn / TimeSeries，统一 `EpiAITrainer.fit()` 接口 |
| **模型注册系统** | `@register` 一行注册新模型，29+ 内置模型 |
| **ModelVault** | 多模型存储、对比表、批量推理 |
| **DeploymentRuntime** | 生产级部署，统一数据表 + 时间连续性检查 + 持久化 |
| **爆发感知损失** | 专门为传染病爆发期设计的损失函数族 |

---

## 适用的预测场景

- 按月/周/日统计的传染病发病数预测
- 多城市/多省份并行建模
- 含外生变量（气象、人口流动）的预测
- 需要同时运行多个模型做对比的业务场景
- 生产环境中持续接收新数据并滚动预测

---

## 三族模型一览

| 族 | Paradigm | 训练方式 | 模型数量 | 代表模型 |
|----|----------|---------|---------|---------|
| Torch | `"torch"` | 梯度下降 + GPU | 10 | CNN, LSTM, Transformer, TimesNet, ResNet, TCN, DLinear, Autoformer, MLP, CNN-LSTM |
| Sklearn | `"sklearn"` | 一次 fit | 6 | RandomForest, XGBoost, LightGBM, SVR, LinearReg, TabPFN |
| TimeSeries | `"ts"` | 统计推断 | 2 | ARIMA, ETS |

---

## 文档索引

| 文档 | 说明 |
|------|------|
| [快速上手](quickstart.md) | 5 分钟跑通完整流程 |
| [框架架构](architecture.md) | 整体分层、设计原则、文件结构 |
| [数据管道](data-pipeline.md) | 加载 / 拆分 / 变换 / 滑窗 |
| [数据模块指南](guides/dataset.md) | 数据模块功能详解 + 添加新 Transform / Split |
| [模型模块指南](guides/models.md) | 注册系统 + 添加新模型 |
| [训练模块指南](guides/training.md) | 训练器三路路由 + 添加新范式 |
| [部署模块指南](guides/deployment.md) | 推理 / 部署 / 扩展 |
| [部署 API 参考](api-deployment.md) | `DeploymentRuntime` 接口说明 |
| [部署设计文档](deployment-design.md) | 部署架构设计细节 |
