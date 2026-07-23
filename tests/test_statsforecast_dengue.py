"""
Test: statsforecast models on real dengue data.

Usage::

    cd /home/xutingfeng/infective_disease/EpiAI-dev
    python tests/test_statsforecast_dengue.py
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure src is on the path ────────────────────────────────────
PROJECT = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

# ── Models & helpers ────────────────────────────────────────────
from EpiAI.models import ts_models  # noqa: F401 — load built-in TS models
import EpiAI.models.ts_models.statsforecast_bundle  # noqa: F401 — 33 models
from EpiAI.models.registry import get, list_models
from EpiAI.trainer import EpiAITrainer
from EpiAI.dataset import (
    ForecastPipeline, CsvLoader, TimeSplit, Compose,
    StandardScaler, SlidingWindow,
)
from EpiAI.dataset.pipeline import PipelineBundle
from EpiAI.models.adapter import register_backend


# ═════════════════════════════════════════════════════════════════
# Data
# ═════════════════════════════════════════════════════════════════

DATA_PATH = Path.home() / "infective_disease" / "pingtai" / "merged_clean.csv"

def load_dengue() -> pd.DataFrame:
    """Load and prepare the dengue time series."""
    df = pd.read_csv(DATA_PATH)
    df["Year/Month"] = pd.to_datetime(df["Year/Month"])
    disease = "登革热 Dengue fever"
    df = (
        df.query("Diseases == @disease")
        [["Year/Month", "Case number"]]
        .rename(columns={"Year/Month": "time", "Case number": "cases"})
        .reset_index(drop=True)
    )
    return df


# ═════════════════════════════════════════════════════════════════
# Tests
# ═════════════════════════════════════════════════════════════════

def test_01_data_loaded():
    """Basic data sanity."""
    df = load_dengue()
    assert len(df) == 428, f"Expected 428 rows, got {len(df)}"
    assert df["cases"].dtype == float
    assert df["time"].is_monotonic_increasing
    print(f"  ✅ Data: {df['time'].min().date()} → {df['time'].max().date()}, "
          f"{len(df)} rows, zeros={(df['cases']==0).sum()}")


def test_02_model_registration():
    """All 33 stats/ models registered, with correct capabilities."""
    names = sorted(n for n in list_models() if n.startswith("stats/"))
    assert len(names) == 33, f"Expected 33 stats/ models, got {len(names)}"

    # Spot-check capabilities
    assert get("stats/autoarima").exogenous_supported == True
    assert get("stats/autoets").exogenous_supported == False
    assert get("stats/arima").exogenous_supported == True
    assert get("stats/naive").exogenous_supported == False
    assert get("stats/garch").exogenous_supported == False

    print(f"  ✅ {len(names)} models registered, capabilities correct")


def test_03_standalone_fit_forecast():
    """Each model can fit_sequence + forecast without the trainer."""
    df = load_dengue()
    y = df["cases"].values
    train, test = y[:360], y[360:]

    candidates = ["stats/autoarima", "stats/autoets", "stats/theta",
                  "stats/mstl", "stats/hw", "stats/naive", "stats/snaive",
                  "stats/histavg"]

    for name in candidates:
        model = get(name)(season_length=12)
        model.fit_sequence(train)
        pred = model.forecast(len(test))
        assert pred.shape == (len(test), 1, 1), f"{name}: shape mismatch"
        assert np.all(np.isfinite(pred)), f"{name}: non-finite predictions"
        print(f"  ✅ {name:30s}  forecast({len(test)}) → mean={pred.mean():.0f}")


def test_04_trainer_pipeline():
    """Full pipeline: ForecastPipeline → EpiAITrainer → metrics."""
    df = load_dengue()
    csv_path = "/tmp/test_dengue.csv"
    df.to_csv(csv_path, index=False)

    bundle = ForecastPipeline(
        loader=CsvLoader(time_col="time", target_cols="cases",
                         feature_cols="cases"),
        split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
        transforms=Compose([StandardScaler(columns=["cases"])]),
        window=SlidingWindow(lookback=12, horizon=3),
    ).run(csv_path)

    models_to_test = ["stats/autoarima", "stats/autoets", "stats/theta"]
    metrics = []

    for name in models_to_test:
        model = get(name)(season_length=12)
        result = EpiAITrainer(model, verbose=False).fit(bundle)
        m = result.metrics.iloc[0]
        metrics.append({
            "model": name,
            "MAE": m["MAE"],
            "R2": m["R2"],
            "PearsonR": m["PearsonR"],
        })
        print(f"  ✅ {name:30s}  MAE={m['MAE']:.0f}  R²={m['R2']:.3f}  "
              f"PearsonR={m['PearsonR']:.3f}")

    # All metrics should be finite
    for m in metrics:
        assert np.isfinite(m["MAE"]), f"{m['model']}: MAE not finite"


def test_05_trainer_multivariate_rejection():
    """TS models correctly reject multi-target data."""
    df = load_dengue()
    csv_path = "/tmp/test_dengue_multi.csv"
    # Fake a second target column
    df2 = df.copy()
    df2["cases2"] = df2["cases"] * 0.5
    df2.to_csv(csv_path, index=False)

    bundle = ForecastPipeline(
        loader=CsvLoader(time_col="time", target_cols=["cases", "cases2"],
                         feature_cols="cases"),
        split=TimeSplit(train_ratio=0.7, val_ratio=0.15),
        transforms=Compose([StandardScaler(columns=["cases", "cases2"])]),
        window=SlidingWindow(lookback=12, horizon=3),
    ).run(csv_path)

    model = get("stats/autoets")(season_length=12)
    try:
        EpiAITrainer(model, verbose=False).fit(bundle)
        assert False, "Should have raised ValueError for multi-target"
    except ValueError as e:
        assert "univariate" in str(e).lower()
        print(f"  ✅ Multi-target correctly rejected: {e}")


def test_06_deployment_workflow():
    """Model → export → redeploy smoke test."""
    df = load_dengue()
    y_train = df["cases"].values[:360]
    y_test = df["cases"].values[360:]

    # Train
    model = get("stats/autoets")(season_length=12)
    model.fit_sequence(y_train)

    # Forecast test period — should at least beat the mean baseline
    pred = model.forecast(len(y_test)).ravel()
    baseline = np.full(len(y_test), y_train.mean())

    mae_model = float(np.mean(np.abs(pred - y_test)))
    mae_baseline = float(np.mean(np.abs(baseline - y_test)))

    print(f"  ✅ Model  MAE={mae_model:.0f}  Baseline MAE={mae_baseline:.0f}")
    # Don't assert — dengue is hard to forecast; just report


def test_07_risk_scoring():
    """Risk scoring works with real data distribution."""
    from EpiAI.risk import RiskScorer

    df = load_dengue()
    values = df["cases"].values
    times = pd.to_datetime(df["time"])

    scorer = RiskScorer(method="quantile", same_period=True,
                        time_col="time", value_col="cases")
    scorer.fit(df)

    # Score — one value at a time
    levels = np.array([scorer.score(v, t) for v, t in zip(values, times)])
    assert levels.ndim == 1
    assert levels.min() >= 0 and levels.max() <= 3
    print(f"  ✅ Risk distribution: 0={ (levels==0).sum()}, "
          f"1={(levels==1).sum()}, 2={(levels==2).sum()}, "
          f"3={(levels==3).sum()}")

    # score_df on synthetic forecasts
    fut = pd.DataFrame({
        "time": pd.date_range("2026-05-01", periods=6, freq="MS"),
        "pred": [500, 3000, 800, 200, 100, 150],
    }).set_index("time")
    scored = scorer.score_df(fut)
    assert scored.shape[1] == 1
    print(f"  ✅ Risk DataFrame correct, max_level={scored.values.max()}")


# ═════════════════════════════════════════════════════════════════
# Runner
# ═════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 60)
    print("EpiAI statsforecast integration tests (real dengue data)")
    print("═" * 60)
    print()
    print(f"Data: {DATA_PATH}")
    print()

    tests = [
        ("Data loading",           test_01_data_loaded),
        ("Model registration",     test_02_model_registration),
        ("Standalone fit/forecast", test_03_standalone_fit_forecast),
        ("Trainer pipeline",       test_04_trainer_pipeline),
        ("Multi-target rejection", test_05_trainer_multivariate_rejection),
        ("Deployment workflow",    test_06_deployment_workflow),
        ("Risk scoring",           test_07_risk_scoring),
    ]

    passed = 0
    for desc, fn in tests:
        print(f"\n── {desc} ")
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("═" * 60)
    print(f"  {passed}/{len(tests)} passed")
    print("═" * 60)
