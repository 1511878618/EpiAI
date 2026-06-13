"""Phase 5 validation: ForecastPipeline (end-to-end orchestrator)."""
import sys, os
src = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, src)

# Bypass __init__.py to avoid torch import
init_py = os.path.join(src, "EpiAI", "dataset", "__init__.py")
bak = init_py + ".bak"
if os.path.exists(init_py):
    os.rename(init_py, bak)

try:
    from EpiAI.dataset.pipeline import ForecastPipeline, PipelineBundle
    from EpiAI.dataset.base import Compose
    from EpiAI.dataset.loaders import CsvLoader
    from EpiAI.dataset.splits import TimeSplit, EntitySplit
    from EpiAI.dataset.transforms import (
        StandardScaler, Log1pTransform, SlidingWindow,
    )
finally:
    if os.path.exists(bak):
        os.rename(bak, init_py)

import numpy as np
import pandas as pd
import tempfile

# ── Build a realistic test CSV ───────────────────────────────────────
np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=100, freq="ME")
df = pd.DataFrame({
    "time": dates,
    "dengue": np.random.lognormal(mean=4, sigma=0.5, size=100).astype(int),
    "flu": np.random.lognormal(mean=5, sigma=0.5, size=100).astype(int),
    "temp": np.random.randn(100) * 5 + 20,
    "humid": np.random.randn(100) * 10 + 60,
})
csv_path = os.path.join(tempfile.gettempdir(), "test_epiai_phase5.csv")
df.to_csv(csv_path, index=False)

# ═════════════════════════════════════════════════════════════════════
# Test 1: Basic pipeline — single entity, time split, no transforms
# ═════════════════════════════════════════════════════════════════════
pipeline = ForecastPipeline(
    loader=CsvLoader(time_col="time", target_cols=["dengue"],
                     feature_cols=["temp", "humid"]),
    split=TimeSplit(train_ratio=0.6, val_ratio=0.2),
    window=SlidingWindow(lookback=6, horizon=3),
)
bundle = pipeline.run(csv_path)

assert isinstance(bundle, PipelineBundle)
assert bundle.n_train > 0
assert bundle.n_val > 0
assert bundle.lookback == 6
assert bundle.horizon == 3
assert bundle.n_features == 2   # temp, humid
assert bundle.n_targets == 1    # dengue
print(f"✓ Test 1: {bundle}")

# ═════════════════════════════════════════════════════════════════════
# Test 2: Pipeline with transforms (Log1p → StandardScaler)
# ═════════════════════════════════════════════════════════════════════
pipeline2 = ForecastPipeline(
    loader=CsvLoader(time_col="time", target_cols=["dengue", "flu"],
                     feature_cols=["temp", "humid"]),
    split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
    transforms=Compose([
        Log1pTransform(columns=["dengue", "flu"]),
        StandardScaler(),
    ]),
    window=SlidingWindow(lookback=12, horizon=4),
)
bundle2 = pipeline2.run(csv_path)
assert bundle2.n_train > 0
assert bundle2.n_targets == 2
assert bundle2.horizon == 4
print(f"✓ Test 2: {bundle2}")

# ═════════════════════════════════════════════════════════════════════
# Test 3: Pipeline without windowing
# ═════════════════════════════════════════════════════════════════════
pipeline3 = ForecastPipeline(
    loader=CsvLoader(time_col="time", target_cols=["dengue"],
                     feature_cols=["temp"]),
    split=TimeSplit(train_ratio=0.6, val_ratio=0.2),
    transforms=Compose([StandardScaler()]),
    window=None,
)
bundle3 = pipeline3.run(csv_path)
assert bundle3.train_x.shape[2] == 1  # one feature
print(f"✓ Test 3 (no window): train={bundle3.n_train}")

# ═════════════════════════════════════════════════════════════════════
# Test 4: Quick method — one-liner
# ═════════════════════════════════════════════════════════════════════
bundle4 = ForecastPipeline.quick(
    path=csv_path,
    time_col="time",
    target_cols="dengue",
    feature_cols=["temp", "humid"],
    lookback=8,
    horizon=2,
    train_ratio=0.7,
    val_ratio=0.15,
    normalize=True,
)
assert bundle4.lookback == 8
assert bundle4.horizon == 2
assert bundle4.n_train > 0
print(f"✓ Test 4 (quick): {bundle4}")

# ═════════════════════════════════════════════════════════════════════
# Test 5: Multi-entity CSV → EntitySplit
# ═════════════════════════════════════════════════════════════════════
df_multi = pd.concat([
    df.assign(city="北京"),
    df.assign(city="上海"),
    df.assign(city="广州"),
]).reset_index(drop=True)
csv_multi = os.path.join(tempfile.gettempdir(), "test_epiai_multi.csv")
df_multi.to_csv(csv_multi, index=False)

pipeline5 = ForecastPipeline(
    loader=CsvLoader(time_col="time", target_cols=["dengue"],
                     feature_cols=["temp", "humid"], entity_col="city"),
    split=EntitySplit(train_entities=["北京", "上海"],
                      val_entities=["广州"], test_entities=[]),
    transforms=Compose([StandardScaler()]),
    window=SlidingWindow(lookback=4, horizon=2),
)
bundle5 = pipeline5.run(csv_multi)
assert bundle5.n_train > 0
assert bundle5.n_val > 0
print(f"✓ Test 5 (multi-entity): train={bundle5.n_train} val={bundle5.n_val}")

# ═════════════════════════════════════════════════════════════════════
# Test 6: Pipeline reproducibility
# ═════════════════════════════════════════════════════════════════════
bundle_a = pipeline.run(csv_path)
bundle_b = pipeline.run(csv_path)
assert np.allclose(bundle_a.train_x, bundle_b.train_x)
assert np.allclose(bundle_a.train_y, bundle_b.train_y)
print("✓ Test 6: Reproducible (deterministic pipeline)")

# ═════════════════════════════════════════════════════════════════════
# Test 7: Test split is optional
# ═════════════════════════════════════════════════════════════════════
pipeline7 = ForecastPipeline(
    loader=CsvLoader(time_col="time", target_cols=["dengue"],
                     feature_cols=["temp"]),
    split=TimeSplit(train_ratio=0.8, val_ratio=0.2),  # no test set
    window=SlidingWindow(lookback=4, horizon=2),
)
bundle7 = pipeline7.run(csv_path)
assert bundle7.test_x is None or len(bundle7.test_x) == 0
print(f"✓ Test 7 (no test split): train={bundle7.n_train} val={bundle7.n_val}")

# ═════════════════════════════════════════════════════════════════════
# Test 8: str target/feature short-hand
# ═════════════════════════════════════════════════════════════════════
bundle8 = ForecastPipeline.quick(
    path=csv_path,
    time_col="time",
    target_cols="dengue",
    feature_cols="temp",
    lookback=3,
    horizon=1,
)
assert bundle8.n_features == 1
assert bundle8.n_targets == 1
print(f"✓ Test 8 (single col): {bundle8}")

# ── Cleanup ─────────────────────────────────────────────────────────
os.unlink(csv_path)
os.unlink(csv_multi)

print("\n🎉 Phase 5: All ForecastPipeline tests passed!")
