"""Phase 4 validation: Transforms and SlidingWindow."""
import sys, os
src = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, src)

# Bypass __init__.py to avoid torch import
init_py = os.path.join(src, "EpiAI", "dataset", "__init__.py")
bak = init_py + ".bak"
if os.path.exists(init_py):
    os.rename(init_py, bak)

try:
    from EpiAI.dataset.transforms import (
        Identity, StandardScaler, RobustScaler, Log1pTransform,
        BoxCoxTransform, SelectColumns, DateFeatures, FeatureLag,
        SlidingWindow, WindowArrays,
    )
    from EpiAI.dataset.base import Compose
finally:
    if os.path.exists(bak):
        os.rename(bak, init_py)

import numpy as np
import pandas as pd

# ── Test data ───────────────────────────────────────────────────────
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=50, freq="ME")
df = pd.DataFrame({
    "time": dates,
    "dengue": np.random.lognormal(mean=4, sigma=0.5, size=50).astype(int),
    "flu": np.random.lognormal(mean=5, sigma=0.5, size=50).astype(int),
    "temp": np.random.randn(50) * 5 + 20,
    "humid": np.random.randn(50) * 10 + 60,
})

train_df = df.iloc[:30].copy()
val_df = df.iloc[30:40].copy()
test_df = df.iloc[40:].copy()

# ═════════════════════════════════════════════════════════════════════
# Test 1: Identity
# ═════════════════════════════════════════════════════════════════════
result = Identity().transform(df)
assert result.equals(df)
print("✓ Identity: OK")

# ═════════════════════════════════════════════════════════════════════
# Test 2: StandardScaler — fit/transform/inverse
# ═════════════════════════════════════════════════════════════════════
scaler = StandardScaler(columns=["dengue", "flu", "temp"])
scaler.fit(train_df)
transformed = scaler.transform(val_df)
# mean should be near 0
assert abs(transformed[["dengue", "flu", "temp"]].mean().max()) < 2.0, \
    f"Mean too large: {transformed[['dengue', 'flu', 'temp']].mean().values}"
# Inverse should restore original
restored = scaler.inverse(transformed)
assert np.allclose(restored[["dengue", "flu", "temp"]].values,
                   val_df[["dengue", "flu", "temp"]].values, rtol=1e-5)
print("✓ StandardScaler fit/transform/inverse: OK")

# ═════════════════════════════════════════════════════════════════════
# Test 3: RobustScaler
# ═════════════════════════════════════════════════════════════════════
rscaler = RobustScaler(columns=["dengue", "flu"])
rscaler.fit(train_df)
rtrans = rscaler.transform(val_df)
rrestored = rscaler.inverse(rtrans)
assert np.allclose(rrestored[["dengue", "flu"]].values,
                   val_df[["dengue", "flu"]].values, rtol=1e-5)
print("✓ RobustScaler fit/transform/inverse: OK")

# ═════════════════════════════════════════════════════════════════════
# Test 4: Log1pTransform
# ═════════════════════════════════════════════════════════════════════
log_t = Log1pTransform(columns=["dengue", "flu"])
logged = log_t.transform(df)
# Values should be smaller than original
assert logged["dengue"].iloc[0] < df["dengue"].iloc[0]
# Inverse should restore
restored = log_t.inverse(logged)
assert np.allclose(restored[["dengue", "flu"]].values,
                   df[["dengue", "flu"]].values, rtol=1e-5)
print("✓ Log1pTransform transform/inverse: OK")

# ═════════════════════════════════════════════════════════════════════
# Test 5: BoxCoxTransform
# ═════════════════════════════════════════════════════════════════════
try:
    bct = BoxCoxTransform(columns=["dengue", "flu"])
    bct.fit(train_df)
    bc_transformed = bct.transform(val_df)
    bc_restored = bct.inverse(bc_transformed)
    assert np.allclose(bc_restored[["dengue", "flu"]].values,
                       val_df[["dengue", "flu"]].values, rtol=1e-3)
    print("✓ BoxCoxTransform fit/transform/inverse: OK")
except ImportError:
    print("  (scipy not installed, skipping BoxCox test)")

# ═════════════════════════════════════════════════════════════════════
# Test 6: SelectColumns
# ═════════════════════════════════════════════════════════════════════
sel = SelectColumns(keep=["dengue", "temp"])
result = sel.transform(df)
assert list(result.columns) == ["dengue", "temp"]
print("✓ SelectColumns(keep): OK")

sel_drop = SelectColumns(drop=["humid"])
result = sel_drop.transform(df)
assert "humid" not in result.columns
assert "dengue" in result.columns
print("✓ SelectColumns(drop): OK")

# ═════════════════════════════════════════════════════════════════════
# Test 7: DateFeatures
# ═════════════════════════════════════════════════════════════════════
dt = DateFeatures(time_col="time", features=["year", "month", "dayofweek", "season"])
result = dt.transform(df)
assert "year" in result.columns
assert "month" in result.columns
assert "dayofweek" in result.columns
assert "season" in result.columns
assert result["year"].iloc[0] == 2020
assert result["season"].iloc[0] in [0, 1, 2, 3]
print("✓ DateFeatures: OK")

# ═════════════════════════════════════════════════════════════════════
# Test 8: FeatureLag
# ═════════════════════════════════════════════════════════════════════
lag = FeatureLag(columns=["dengue"], lags=[1, 3], drop_na=True)
result = lag.transform(df)
assert "dengue_lag_1" in result.columns
assert "dengue_lag_3" in result.columns
assert len(result) == len(df) - 3  # dropped NaN rows
print(f"✓ FeatureLag: shape {df.shape} → {result.shape}")

# ═════════════════════════════════════════════════════════════════════
# Test 9: Compose — multiple transforms
# ═════════════════════════════════════════════════════════════════════
pipe = Compose([
    Log1pTransform(columns=["dengue", "flu"]),
    StandardScaler(columns=["dengue", "flu", "temp"]),
])
pipe.fit(train_df)
result = pipe.transform(val_df)
assert abs(result["temp"].mean()) < 2.0  # standardized
# Inverse full pipeline
restored = pipe.inverse(result)
assert np.allclose(restored[["dengue", "flu"]].values,
                   val_df[["dengue", "flu"]].values, rtol=1e-4)
print("✓ Compose(Log→Standard): OK")

# ═════════════════════════════════════════════════════════════════════
# Test 10: SlidingWindow — single entity
# ═════════════════════════════════════════════════════════════════════
sw = SlidingWindow(lookback=4, horizon=2)
windows = sw.apply(df, target_cols=["dengue"], feature_cols=["temp", "humid"])
assert windows.x.shape == (len(df) - 4 - 2 + 1, 4, 2), f"Got {windows.x.shape}"
assert windows.y.shape == (len(df) - 4 - 2 + 1, 2, 1), f"Got {windows.y.shape}"
print(f"✓ SlidingWindow(single): x={windows.x.shape} y={windows.y.shape}")

# ═════════════════════════════════════════════════════════════════════
# Test 11: SlidingWindow — multi entity
# ═════════════════════════════════════════════════════════════════════
df_multi = pd.concat([
    df.assign(city="北京"),
    df.assign(city="上海"),
]).reset_index(drop=True)
sw = SlidingWindow(lookback=3, horizon=1, stride=1)
windows = sw.apply(df_multi, target_cols=["dengue"],
                   feature_cols=["temp"], entity_col="city")
expected = 2 * (len(df) - 3 - 1 + 1)  # 2 cities
assert windows.x.shape[0] == expected, f"Expected {expected}, got {windows.x.shape[0]}"
print(f"✓ SlidingWindow(multi): {windows.x.shape[0]} windows")

# ═════════════════════════════════════════════════════════════════════
# Test 12: WindowArrays repr
# ═════════════════════════════════════════════════════════════════════
wa = WindowArrays(x=np.zeros((5, 3, 2)), y=np.zeros((5, 1, 1)),
                  feature_names=["a", "b"], target_names=["t"])
assert "5, 3, 2" in repr(wa)
print("✓ WindowArrays repr: OK")

# ═════════════════════════════════════════════════════════════════════
# Test 13: SlidingWindow stride > 1
# ═════════════════════════════════════════════════════════════════════
sw2 = SlidingWindow(lookback=4, horizon=2, stride=3)
w2 = sw2.apply(df, target_cols=["dengue"], feature_cols=["temp"])
assert w2.x.shape[0] < windows.x.shape[0]  # fewer windows with stride=3
print(f"✓ SlidingWindow(stride=3): {w2.x.shape[0]} windows (vs {windows.x.shape[0]} with stride=1)")

# ═════════════════════════════════════════════════════════════════════
# Test 14: SlidingWindow ahead > 0
# ═════════════════════════════════════════════════════════════════════
sw3 = SlidingWindow(lookback=4, horizon=2, ahead=1)
w3 = sw3.apply(df, target_cols=["dengue"], feature_cols=["temp"])
assert w3.x.shape[0] < windows.x.shape[0]  # gap reduces windows
print(f"✓ SlidingWindow(ahead=1): {w3.x.shape[0]} windows")

# ═════════════════════════════════════════════════════════════════════
# Test 15: end-to-end pipeline sample
# ═════════════════════════════════════════════════════════════════════
pipeline = Compose([
    Log1pTransform(columns=["dengue", "flu"]),
    StandardScaler(columns=["dengue", "flu", "temp", "humid"]),
    DateFeatures(time_col="time", features=["month"]),
])
pipeline.fit(train_df)
processed = pipeline.transform(train_df)
sw = SlidingWindow(lookback=6, horizon=3)
windows = sw.apply(processed, target_cols=["dengue"],
                   feature_cols=[c for c in processed.columns
                                if c not in ["dengue", "time"]])
assert len(windows.x) > 0
assert windows.x.shape[1] == 6   # lookback
assert windows.y.shape[1] == 3   # horizon
print(f"✓ End-to-end pipeline: x={windows.x.shape} y={windows.y.shape}")

print("\n🎉 Phase 4: All Transform tests passed!")
