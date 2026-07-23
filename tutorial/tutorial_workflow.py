"""
EpiAI 极简流程：训练 → 部署 → 保存 → 服务器加载预测
====================================================

Usage::

    cd /home/xutingfeng/infective_disease/EpiAI-dev

    # 若训练并保存模型:
    python tutorial/tutorial_workflow.py --train

    # 若在服务器加载已保存模型并预测:
    python tutorial/tutorial_workflow.py --predict
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from EpiAI import ForecastTask


# ═════════════════════════════════════════════════════════════════
# 配置（按需修改）
# ═════════════════════════════════════════════════════════════════

DATA_PATH = Path.home() / "infective_disease" / "pingtai" / "merged_clean.csv"
DISEASE = "登革热 Dengue fever"
RUNTIME_PATH = "/tmp/epiai_dengue_runtime"  # 模型保存路径（服务器上可改为持久化路径）


def load_data() -> pd.DataFrame:
    """加载并预处理数据。"""
    df = pd.read_csv(DATA_PATH)
    df["Year/Month"] = pd.to_datetime(df["Year/Month"])
    df = (
        df.query("Diseases == @DISEASE")
        [["Year/Month", "Case number"]]
        .rename(columns={"Year/Month": "time", "Case number": "cases"})
        .reset_index(drop=True)
    )
    # 只取 2010 年以后的数据（登革热爆发主要在近年）
    df = df[df["time"].dt.year >= 2010].reset_index(drop=True)
    return df


# ═════════════════════════════════════════════════════════════════
# 流程一：训练 → 部署 → 保存
# ═════════════════════════════════════════════════════════════════

def train_and_save():
    """训练模型、部署、保存到磁盘。"""
    print("=" * 50)
    print("Step 1: 加载数据")
    print("=" * 50)
    df = load_data()
    print(f"  {len(df)} 条记录, {df['time'].min().date()} → {df['time'].max().date()}")
    print()

    print("=" * 50)
    print("Step 2: 定义预测任务")
    print("=" * 50)
    task = ForecastTask(
        data=df,
        target="cases",
        time_col="time",
        freq="MS",          # 月数据
        horizon=12,         # 预测未来 12 个月
    )
    print(f"  freq={task.freq}, season_length={task.freq_to_season_length()}")
    print()

    print("=" * 50)
    print("Step 3: 批量训练 + 对比排行")
    print("=" * 50)
    vault = task.compare(["sf_auto_ets", "sf_holt_winters", "sf_theta"])
    print()
    print("排行榜（按 Pearson R 降序）:")
    print(vault.summary().to_string())
    print()

    print("=" * 50)
    print("Step 4: 部署最佳模型 + 风险评分")
    print("=" * 50)
    best_name = vault.best("R2")
    print(f"  最佳模型: {best_name}")
    runtime = task.deploy(best_name, vault, risk="quantile")
    print(f"  Runtime: {runtime}")
    print()

    print("=" * 50)
    print("Step 5: 试预测")
    print("=" * 50)
    report = runtime.predict(horizon=6)
    print(report.to_string())
    print()

    print("=" * 50)
    print("Step 6: 保存模型到磁盘")
    print("=" * 50)
    saved_path = runtime.save(RUNTIME_PATH)
    print(f"  已保存至: {saved_path}")
    print(f"  目录内容: {list(Path(saved_path).iterdir())}")
    print()

    print("✅ 训练完成。模型已保存。")
    print(f"   服务器上加载命令: python {__file__} --predict")


# ═════════════════════════════════════════════════════════════════
# 流程二：服务器加载 → 预测
# ═════════════════════════════════════════════════════════════════

def load_and_predict():
    """在服务器加载已保存的模型、接收新数据、预测。"""
    from EpiAI.inference import DeploymentRuntime

    print("=" * 50)
    print("服务器: 加载已保存的模型")
    print("=" * 50)
    runtime = DeploymentRuntime.load(RUNTIME_PATH)
    print(f"  Runtime: {runtime}")
    print(f"  模型: {list(runtime.vault.models.keys())}")
    print(f"  历史数据: {len(runtime.data_table)} 行")
    print()

    print("=" * 50)
    print("服务器: 最新数据预测 6 期")
    print("=" * 50)
    report = runtime.predict(horizon=6)
    print(report.to_string())
    print()

    print("=" * 50)
    print("服务器: 模拟新数据到达 → 追加 → 再预测")
    print("=" * 50)
    # 模拟 2026 年 5 月的新观测
    new_row = pd.DataFrame({"time": [pd.Timestamp("2026-05-01")], "cases": [150.0]})
    runtime.feed(new_row)
    print(f"  已追加新数据，当前 {len(runtime.data_table)} 行")
    print()

    report2 = runtime.predict(horizon=6)
    print(report2.to_string())
    print()

    print("✅ 服务器预测完成。")


# ═════════════════════════════════════════════════════════════════
# 入口
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EpiAI 一站式流程")
    parser.add_argument("--train", action="store_true", help="训练并保存模型")
    parser.add_argument("--predict", action="store_true", help="加载模型并预测")
    args = parser.parse_args()

    if args.predict:
        load_and_predict()
    else:
        # 默认执行训练（含 --train 或没有参数）
        train_and_save()
