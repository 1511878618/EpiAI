# Bug 汇总

> 已知问题、复现条件、影响范围。

---

## FeatureLag + 各 split 独立 transform 导致窗口模型丢失数据

**ID:** BUG-001
**状态:** 未修复
**发现日期:** 2026-06-16

### 问题

`ForecastPipeline` 对 val/test split 先做 Transform（含 `FeatureLag`），再拼接上下文。`FeatureLag` 在短 split 上生成 NaN，导致后续窗口中输入含 NaN，部分窗口被丢弃。

### 复现条件

```python
transforms=Compose([
    FeatureLag(columns=["cases"], lags=[1, 2, 3, 12]),
]),
window=SlidingWindow(lookback=12, horizon=3),
```

### 影响

窗口模型的有效测试窗口数 = `n_test - max(lags)`。例中 lags=12 时 n=20（应 n=32）。

### 根因

```python
# pipeline.py 中 transform 先于 context 拼接
val_df = self.transforms.transform(val_df)     # ← 先 transform
test_df = self.transforms.transform(test_df)   # ← 先 transform
# ... 然后才
_test_ctx = _prev_for_test.iloc[-context_len:]
test_w = self.window.apply(pd.concat([_test_ctx, test_df]), ...)
```

`FeatureLag` 需要历史数据来生成滞后特征，但它在短 split 上独立执行，没有历史。

### 修复方向

- 方案 A：`FeatureLag` 执行前先为当前 split 拼接前一个 split 的尾部
- 方案 B：将 transform 的执行时机移到 context 拼接之后
- 方案 C：`FeatureLag` 内部对短输入自动 padding

### 是否 blocking

否。时序模型（ARIMA/ETS）不受影响。仅 n 值偏小，指标仍可参考。
