"""Phase 1 validation: data container and base abstractions."""
import sys, os
src = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, src)

# We need to import container + base without __init__.py side-effects.
# Temporarily rename __init__.py so torch import doesn't fire.
init_py = os.path.join(src, "EpiAI", "dataset", "__init__.py")
bak = init_py + ".bak"
if os.path.exists(init_py):
    os.rename(init_py, bak)

try:
    from EpiAI.dataset.container import TimeSeriesData, SplitResult, WindowBundle
    from EpiAI.dataset.base import DataLoader, SplitStrategy, Transform, Compose
finally:
    if os.path.exists(bak):
        os.rename(bak, init_py)

import numpy as np
import pandas as pd

# ── Test TimeSeriesData ──────────────────────────────────────────────
dates = pd.date_range("2020-01", periods=10, freq="ME")
df = pd.DataFrame({
    "time": dates,
    "city": ["北京"]*5 + ["上海"]*5,
    "dengue": np.random.randint(0, 100, 10),
    "flu": np.random.randint(0, 500, 10),
    "temp": np.random.randn(10),
    "humid": np.random.randn(10),
})

# Single entity
d1 = TimeSeriesData(df=df.drop(columns=["city"]), time_col="time",
                    target_cols=["dengue"], feature_cols=["temp"])
assert d1.n_entities == 1, f"Expected 1, got {d1.n_entities}"
print("✓ Single entity: n_entities =", d1.n_entities)

# Multi entity
d2 = TimeSeriesData(df=df, time_col="time", target_cols=["dengue"],
                    feature_cols=["temp"], entity_col="city")
assert d2.n_entities == 2
assert d2.entity_values == ["上海", "北京"]  # sorted
print("✓ Multi entity:", d2.entity_values)

# Time range
assert d1.time_range[0] == dates[0]
assert d1.time_range[1] == dates[-1]
print("✓ Time range:", d1.time_range)

# ── Test SplitResult ────────────────────────────────────────────────
sr = SplitResult(train_idx=np.array([0,1,2]), val_idx=np.array([3,4]),
                 test_idx=np.array([5,6,7,8,9]))
assert sr.n_train == 3 and sr.n_val == 2 and sr.n_test == 5
print("✓ SplitResult: train=%d val=%d test=%d" % (sr.n_train, sr.n_val, sr.n_test))

# ── Test WindowBundle ────────────────────────────────────────────────
wb = WindowBundle(train_x=np.zeros((10, 12, 5)), train_y=np.zeros((10, 3, 2)))
assert wb.train_x.shape == (10, 12, 5)
print("✓ WindowBundle:", wb.train_x.shape)

# ── Test base classes ────────────────────────────────────────────────
assert hasattr(DataLoader, "load")
assert hasattr(SplitStrategy, "split")
assert hasattr(Transform, "transform")
print("✓ Abstract base classes: DataLoader, SplitStrategy, Transform")

# ── Test Compose ──────────────────────────────────────────────────────
class Double(Transform):
    def transform(self, df):
        return df * 2

class AddOne(Transform):
    def transform(self, df):
        return df + 1

pipe = Compose([Double(), AddOne()])
result = pipe.transform(pd.DataFrame({"a": [1, 2, 3]}))
assert result["a"].tolist() == [3, 5, 7], f"Got {result['a'].tolist()}"
print("✓ Compose: (x*2)+1 =", result["a"].tolist())

print("\n🎉 Phase 1: All tests passed!")
