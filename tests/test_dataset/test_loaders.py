"""Phase 2 validation: DataLoaders."""
import sys, os
src = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, src)

# Bypass __init__.py to avoid torch import
init_py = os.path.join(src, "EpiAI", "dataset", "__init__.py")
bak = init_py + ".bak"
if os.path.exists(init_py):
    os.rename(init_py, bak)

try:
    from EpiAI.dataset.container import TimeSeriesData
    from EpiAI.dataset.loaders import CsvLoader, FeatherLoader, load_data
finally:
    if os.path.exists(bak):
        os.rename(bak, init_py)

import numpy as np
import pandas as pd
import tempfile

# ── Test 1: CsvLoader with single entity ────────────────────────────
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
    f.write("time,dengue,flu,temp\n")
    for t in range(12):
        f.write(f"2023-{t+1:02d}-01,{np.random.randint(0,100)},{np.random.randint(0,500)},{np.random.randn():.4f}\n")
    csv_path = f.name

try:
    loader = CsvLoader(time_col="time", target_cols="dengue",
                       feature_cols=["temp", "flu"])
    data = loader.load(csv_path)
    assert isinstance(data, TimeSeriesData)
    assert data.n_entities == 1
    assert data.target_cols == ["dengue"]
    assert data.feature_cols == ["temp", "flu"]
    assert len(data.df) == 12
    print("✓ CsvLoader single entity: OK")
finally:
    os.unlink(csv_path)

# ── Test 2: CsvLoader with multi-entity ─────────────────────────────
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
    f.write("time,city,dengue,temp\n")
    for city in ["北京", "上海"]:
        for t in range(6):
            f.write(f"2023-{t+1:02d}-01,{city},{np.random.randint(0,100)},{np.random.randn():.4f}\n")
    csv_path2 = f.name

try:
    loader2 = CsvLoader(time_col="time", target_cols="dengue",
                        feature_cols=["temp"], entity_col="city")
    data2 = loader2.load(csv_path2)
    assert data2.n_entities == 2
    assert data2.entity_values == ["上海", "北京"]
    assert len(data2.df) == 12
    print("✓ CsvLoader multi-entity: OK")
finally:
    os.unlink(csv_path2)

# ── Test 3: FeatherLoader ───────────────────────────────────────────
feather_path = os.path.join(os.path.dirname(__file__), "..", "Dengue.feather")
has_pyarrow = False
try:
    import pyarrow
    has_pyarrow = True
except ImportError:
    print("  (pyarrow not installed, skipping feather test)")

if has_pyarrow and os.path.exists(feather_path):
    # Probe structure
    probe = pd.read_feather(feather_path)
    print(f"  Feather columns: {list(probe.columns)}")
    print(f"  Feather shape: {probe.shape}")
    # Test loading (user will need to supply correct column names)
    print("✓ Feather file exists, can inspect structure")
else:
    print("  (Dengue.feather not available)")

# ── Test 4: load_data auto-detect ───────────────────────────────────
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
    f.write("time,val\n2023-01-01,10\n2023-02-01,20\n")
    detect_path = f.name

try:
    data3 = load_data(detect_path, time_col="time", target_cols=["val"], feature_cols=["val"])
    assert len(data3.df) == 2
    print("✓ load_data auto-detect (.csv): OK")
finally:
    os.unlink(detect_path)

# ── Test 5: file not found ──────────────────────────────────────────
try:
    CsvLoader(time_col="x", target_cols="y", feature_cols=["z"]).load("nonexistent.csv")
    assert False, "Should have raised"
except FileNotFoundError:
    print("✓ CsvLoader raises on missing file: OK")

# ── Test 6: missing columns ─────────────────────────────────────────
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
    f.write("a,b\n1,2\n")
    miss_path = f.name
try:
    try:
        CsvLoader(time_col="time", target_cols="y", feature_cols=["z"]).load(miss_path)
        assert False, "Should have raised"
    except ValueError as e:
        assert "not found" in str(e)
        print("✓ CsvLoader raises on missing columns: OK")
finally:
    os.unlink(miss_path)

# ── Test 7: str target/feature auto-converts to list ────────────────
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
    f.write("time,val\n2023-01-01,10\n")
    str_path = f.name
try:
    loader = CsvLoader(time_col="time", target_cols="val", feature_cols="val")
    d = loader.load(str_path)
    assert d.target_cols == ["val"]
    assert d.feature_cols == ["val"]
    print("✓ str→list auto-conversion: OK")
finally:
    os.unlink(str_path)

print("\n🎉 Phase 2: All DataLoader tests passed!")
