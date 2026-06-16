"""Phase 3 validation: SplitStrategy implementations."""
import sys, os
src = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, src)

# Bypass __init__.py to avoid torch import
init_py = os.path.join(src, "EpiAI", "dataset", "__init__.py")
bak = init_py + ".bak"
if os.path.exists(init_py):
    os.rename(init_py, bak)

try:
    from EpiAI.dataset.container import TimeSeriesData, SplitResult
    from EpiAI.dataset.splits import (
        TimeSplit, EntitySplit, EntityTimeSplit,
        CustomIndexSplit, NoSplit, CrossValidationSplit,
    )
finally:
    if os.path.exists(bak):
        os.rename(bak, init_py)

import numpy as np
import pandas as pd

# ── Shared test data ────────────────────────────────────────────────
dates = pd.date_range("2020-01-01", periods=30, freq="ME")
df_single = pd.DataFrame({
    "time": dates,
    "dengue": np.random.randint(0, 200, 30),
    "temp": np.random.randn(30),
})

df_multi = pd.DataFrame({
    "time": list(dates) * 2,
    "city": ["北京"] * 30 + ["上海"] * 30,
    "dengue": np.random.randint(0, 200, 60),
    "temp": np.random.randn(60),
})

data_single = TimeSeriesData(
    df=df_single, time_col="time",
    target_cols=["dengue"], feature_cols=["temp"],
)

data_multi = TimeSeriesData(
    df=df_multi, time_col="time",
    target_cols=["dengue"], feature_cols=["temp"],
    entity_col="city",
)


# ═════════════════════════════════════════════════════════════════════
# Test 1: TimeSplit with explicit dates
# ═════════════════════════════════════════════════════════════════════
result = TimeSplit(train_end="2021-01-01", val_end="2021-07-01").split(data_single)
assert len(result.train_idx) > 0, "train should not be empty"
assert result.n_train + result.n_val + result.n_test == 30
print(f"✓ TimeSplit(date): train={result.n_train} val={result.n_val} test={result.n_test}")

# ═════════════════════════════════════════════════════════════════════
# Test 2: TimeSplit with ratios
# ═════════════════════════════════════════════════════════════════════
result = TimeSplit(train_ratio=0.6, val_ratio=0.2).split(data_single)
assert result.n_train == 18, f"Expected 18 train, got {result.n_train}"
assert result.n_val == 6, f"Expected 6 val, got {result.n_val}"
assert result.n_test == 6, f"Expected 6 test, got {result.n_test}"
print(f"✓ TimeSplit(ratio): train={result.n_train} val={result.n_val} test={result.n_test}")

# ═════════════════════════════════════════════════════════════════════
# Test 3: EntitySplit
# ═════════════════════════════════════════════════════════════════════
result = EntitySplit(
    train_entities=["北京"], val_entities=["上海"], test_entities=[]
).split(data_multi)
assert result.n_train == 30
assert result.n_val == 30
assert result.n_test == 0
print(f"✓ EntitySplit: train={result.n_train} val={result.n_val}")

# ═════════════════════════════════════════════════════════════════════
# Test 4: EntityTimeSplit — each city has its own time window
# ═════════════════════════════════════════════════════════════════════
result = EntityTimeSplit({
    "北京": ("2020-07-01", "2020-10-01"),
    "上海": ("2020-07-01", "2020-10-01"),
}).split(data_multi)
assert result.n_train > 0
assert result.n_val > 0
assert result.n_test > 0
print(f"✓ EntityTimeSplit: train={result.n_train} val={result.n_val} test={result.n_test}")

# ═════════════════════════════════════════════════════════════════════
# Test 5: CustomIndexSplit
# ═════════════════════════════════════════════════════════════════════
result = CustomIndexSplit(
    train_idx=[0, 1, 2, 3, 4],
    val_idx=[5, 6, 7],
    test_idx=[8, 9],
).split(data_single)
assert result.n_train == 5
assert result.n_val == 3
assert result.n_test == 2
print(f"✓ CustomIndexSplit: train={result.n_train} val={result.n_val} test={result.n_test}")

# ═════════════════════════════════════════════════════════════════════
# Test 6: CustomIndexSplit — out of range raises
# ═════════════════════════════════════════════════════════════════════
try:
    CustomIndexSplit(train_idx=[0, 1, 999], val_idx=[], test_idx=[]).split(data_single)
    assert False, "Should have raised"
except ValueError:
    print("✓ CustomIndexSplit out-of-range: caught correctly")

# ═════════════════════════════════════════════════════════════════════
# Test 7: NoSplit
# ═════════════════════════════════════════════════════════════════════
result = NoSplit().split(data_single)
assert result.n_train == 30
assert result.n_val == 0
assert result.n_test == 0
print(f"✓ NoSplit: train={result.n_train}")

# ═════════════════════════════════════════════════════════════════════
# Test 8: CrossValidationSplit — first fold & folds generator
# ═════════════════════════════════════════════════════════════════════
cv = CrossValidationSplit(n_splits=5, val_horizon=4)
result = cv.split(data_single)
assert result.n_train > 0
assert result.n_val > 0
print(f"✓ CrossValidationSplit (first fold): train={result.n_train} val={result.n_val}")

fold_count = 0
for fold_result in cv.folds(data_single):
    assert fold_result.n_train > 0
    assert fold_result.n_val > 0
    fold_count += 1
assert fold_count == 5
print(f"✓ CrossValidationSplit folds: {fold_count} folds")

# ═════════════════════════════════════════════════════════════════════
# Test 9: EntitySplit without entity_col raises
# ═════════════════════════════════════════════════════════════════════
try:
    EntitySplit(train_entities=["X"], val_entities=["Y"], test_entities=[]).split(data_single)
    assert False, "Should have raised"
except ValueError:
    print("✓ EntitySplit without entity_col: caught correctly")

# ═════════════════════════════════════════════════════════════════════
# Test 10: TimeSplit — train_idx should be strictly before val_idx
# ═════════════════════════════════════════════════════════════════════
result = TimeSplit(train_end="2020-06-01", val_end="2020-10-01").split(data_single)
assert max(result.train_idx) < min(result.val_idx), \
    f"train max {max(result.train_idx)} >= val min {min(result.val_idx)}"
assert max(result.val_idx) < min(result.test_idx), \
    f"val max {max(result.val_idx)} >= test min {min(result.test_idx)}"
print("✓ TimeSplit temporal ordering: correct")

# ═════════════════════════════════════════════════════════════════════
# Test 11: Cumulative counts match total
# ═════════════════════════════════════════════════════════════════════
for name, strategy in [
    ("TimeSplit(ratio)", TimeSplit(train_ratio=0.7, val_ratio=0.15)),
    ("EntitySplit", EntitySplit(["北京"], ["上海"], [])),
    ("NoSplit", NoSplit()),
]:
    r = strategy.split(data_multi if "Entity" in name else data_single)
    total = r.n_train + r.n_val + r.n_test
    original = len(data_multi.df) if "Entity" in name else len(data_single.df)
    assert total == original, f"{name}: {total} != {original}"
print("✓ All splits cover full dataset")

print("\n🎉 Phase 3: All SplitStrategy tests passed!")
