"""Phase 6: Integration validation — new pipeline import + end-to-end demo."""
import sys, os
src = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, src)

# ═════════════════════════════════════════════════════════════════════
# Test 1: Import new pipeline WITHOUT torch
# ═════════════════════════════════════════════════════════════════════
# Deliberately NOT importing torch — the new pipeline should not need it.
from EpiAI.dataset import (
    TimeSeriesData, SplitResult, WindowBundle,
    CsvLoader, TimeSplit, EntitySplit, CustomIndexSplit, NoSplit,
    StandardScaler, RobustScaler, Log1pTransform, Identity,
    SelectColumns, DateFeatures, FeatureLag,
    SlidingWindow, WindowArrays,
    ForecastPipeline, PipelineBundle,
    Compose, Transform, DataLoader, SplitStrategy,
)
print("✓ Test 1: New pipeline imports fine without torch")

# ═════════════════════════════════════════════════════════════════════
# Test 2: Build & save a sample CSV
# ═════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import tempfile

np.random.seed(42)
dates = pd.date_range("2020-01-01", periods=60, freq="ME")
df = pd.DataFrame({
    "time": dates,
    "dengue": np.random.lognormal(mean=4, sigma=0.5, size=60).astype(int),
    "flu": np.random.lognormal(mean=5, sigma=0.5, size=60).astype(int),
    "temp": np.random.randn(60) * 5 + 20,
})
csv_path = os.path.join(tempfile.gettempdir(), "epiai_integration_test.csv")
df.to_csv(csv_path, index=False)

# ═════════════════════════════════════════════════════════════════════
# Test 3: Quick pipeline — full end-to-end
# ═════════════════════════════════════════════════════════════════════
bundle = ForecastPipeline.quick(
    path=csv_path,
    time_col="time",
    target_cols=["dengue", "flu"],
    feature_cols=["temp"],
    lookback=6,
    horizon=2,
    normalize=True,
)
assert bundle.n_train > 0
assert bundle.n_val > 0
print(f"✓ Test 3: Quick pipeline → train={bundle.n_train} val={bundle.n_val} "
      f"features={bundle.n_features} targets={bundle.n_targets}")

# ═════════════════════════════════════════════════════════════════════
# Test 4: Manual pipeline — full control
# ═════════════════════════════════════════════════════════════════════
pipeline = ForecastPipeline(
    loader=CsvLoader(time_col="time", target_cols=["dengue"],
                     feature_cols=["temp"]),
    split=TimeSplit(train_ratio=0.6, val_ratio=0.2),
    transforms=Compose([
        Log1pTransform(columns=["dengue"]),
        StandardScaler(),
    ]),
    window=SlidingWindow(lookback=4, horizon=1),
)
bundle2 = pipeline.run(csv_path)
assert bundle2.n_train > 0
assert bundle2.n_val > 0
# Check that transforms were applied (values should be centered)
train_mean = bundle2.train_x.mean()
assert abs(train_mean) < 1.0, f"Train x mean should be near 0, got {train_mean:.4f}"
print(f"✓ Test 4: Manual pipeline → train={bundle2.n_train} "
      f"x_mean={train_mean:.4f}")

# ═════════════════════════════════════════════════════════════════════
# Test 5: Pipeline integrity — shapes match expectations
# ═════════════════════════════════════════════════════════════════════
assert bundle2.train_x.ndim == 3, f"Expected 3D, got {bundle2.train_x.ndim}D"
assert bundle2.train_x.shape[1] == 4   # lookback
assert bundle2.train_y.shape[1] == 1   # horizon
assert bundle2.train_x.shape[2] == 1   # n_features
assert bundle2.train_y.shape[2] == 1   # n_targets
print(f"✓ Test 5: Shape integrity — {bundle2.train_x.shape} → {bundle2.train_y.shape}")

# ═════════════════════════════════════════════════════════════════════
# Test 6: Legacy pipeline — lazy load (requires torch to be installable)
# ═════════════════════════════════════════════════════════════════════
try:
    from EpiAI.dataset import _load_legacy
    _load_legacy()
    from EpiAI.dataset import MultiTargetCityDatasetBuilder, DatasetConfig, ForecastDataModule
    print("✓ Test 6: Legacy pipeline loaded (torch available)")
except ImportError:
    print("  (torch not installed, legacy pipeline skipped)")

# ── Cleanup ─────────────────────────────────────────────────────────
os.unlink(csv_path)

print("\n🎉 Phase 6: Integration validation passed!")
