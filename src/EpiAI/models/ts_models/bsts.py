"""
Bayesian Structural Time Series (BSTS) forecaster.

Implements the same model as R's ``bsts`` package:
  y_t = μ_t + γ_t + ε_t
  μ_t = μ_{t-1} + η_t        (local level, random walk)
  γ_t = seasonal effect       (12-month seasonal, sum-to-zero)

Uses PyMC for MCMC sampling (matching R's niter + burn behavior).
Falls back to statsmodels MLE when PyMC is unavailable.

Reference:
  - R's bsts package: AddLocalLevel + AddSeasonal, niter=250, burn=50
  - Scott & Varian (2014), "Predicting the present with Bayesian structural
    time series"
"""
from __future__ import annotations

import numpy as np

from EpiAI.models.base import TSMixin
from EpiAI.models.registry import register

try:
    import pymc as pm
    import pytensor.tensor as pt
    _HAS_PYMC = True
except ImportError:
    _HAS_PYMC = False


@register("BSTS")
class BSTSForecaster(TSMixin):
    """Bayesian Structural Time Series (local level + seasonal).

    Mirrors R's ``bsts::AddLocalLevel() + AddSeasonal()`` with MCMC sampling.

    Parameters
    ----------
    seasonal_periods : int, default=12
        Number of seasonal periods (e.g. 12 for monthly).
    niter : int, default=250
        Number of MCMC draws after burn-in (matching R's ``niter``).
    burn : int, default=50
        Number of burn-in / tuning samples (matching R's ``burn``).
    clip_negative : bool, default=True
        Clip negative predictions to zero.
    chains : int, default=1
        Number of MCMC chains.
    """

    def __init__(
        self,
        seasonal_periods: int = 12,
        niter: int = 250,
        burn: int = 50,
        clip_negative: bool = True,
        chains: int = 1,
        **kwargs,
    ) -> None:
        self.seasonal_periods = seasonal_periods
        self.niter = niter
        self.burn = burn
        self.clip_negative = clip_negative
        self.chains = chains

        self._trace = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._n_train: int = 0
        self._y_train: np.ndarray | None = None

        # statsmodels fallback
        self._ss_result = None

    def _to_1d(self, y):
        y = np.asarray(y, dtype=float)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        return y

    # ── Fit ─────────────────────────────────────────────────────

    def fit_sequence(self, y_train, X_train=None, **kwargs):
        """Fit BSTS with MCMC.

        Parameters
        ----------
        y_train : array-like, shape (n,)
        X_train : ignored
        """
        y = self._to_1d(y_train)
        self._y_train = y.copy()
        self._n_train = len(y)

        if _HAS_PYMC:
            self._fit_mcmc(y)
        else:
            self._fit_mle(y)
        return self

    def _fit_mcmc(self, y):
        """PyMC MCMC fit — local level + sum-to-zero seasonal."""
        n = len(y)
        m = self.seasonal_periods
        self._y_mean = float(np.mean(y))
        self._y_std = float(np.std(y)) or 1.0
        y_s = (y - self._y_mean) / self._y_std

        with pm.Model() as model:
            σ_l = pm.HalfCauchy("sigma_level", 0.5)
            σ_s = pm.HalfCauchy("sigma_seasonal", 0.5)
            σ_o = pm.HalfCauchy("sigma_obs", 0.5)

            level = pm.GaussianRandomWalk("level", sigma=σ_l, shape=n)

            # Seasonal: 11 free params → 12th = -sum, sum(12) = 0
            s_raw = pm.Normal("s", mu=0, sigma=σ_s, shape=m - 1)
            s_last = -pt.sum(s_raw, keepdims=True)
            s_full = pt.concatenate([s_raw, s_last], axis=0)
            seasonal = s_full[pm.math.arange(n) % m]

            pm.Normal("obs", mu=level + seasonal, sigma=σ_o, observed=y_s)

            self._trace = pm.sample(
                draws=self.niter, tune=self.burn,
                chains=self.chains, cores=1,
                random_seed=42, progressbar=False,
            )

    def _fit_mle(self, y):
        """Fallback: statsmodels MLE."""
        from statsmodels.tsa.statespace.structural import \
            UnobservedComponents
        freq = self.seasonal_periods
        ss_model = UnobservedComponents(
            y, level=True,
            seasonal=freq if freq > 1 else 0)
        self._ss_result = ss_model.fit(method="powell", disp=False)

    # ── Prediction (MCMC projection, matching R's predict) ────

    def _project_forward(self, n: int) -> np.ndarray:
        """Project posterior forward from MCMC trace.

        For each posterior draw:
          1. Take last level value: level[-1]
          2. Random walk forward n steps with sigma_level[j]
          3. Add seasonal effects
          4. Add observation noise
        Aggregate across draws → posterior predictive mean.

        This matches R's ``predict(bsts_model, horizon=n, burn=B)$mean``.
        """
        tr = self._trace
        draws = tr.posterior["level"].values          # (chain, draw, n_train)
        σ_l = tr.posterior["sigma_level"].values      # (chain, draw)
        σ_o = tr.posterior["sigma_obs"].values        # (chain, draw)
        s = tr.posterior["s"].values                  # (chain, draw, 11)

        n_ch, n_d, _ = draws.shape
        n_fcst = n
        n_train = self._n_train
        m = self.seasonal_periods

        # Build full seasonal (12 effects from 11)
        s_full = np.concatenate(
            [s, -s.sum(axis=-1, keepdims=True)], axis=-1)  # sum=0

        rng = np.random.default_rng(42)
        fcst_samples = np.zeros((n_ch * n_d, n_fcst))

        for i in range(n_ch):
            for j in range(n_d):
                idx = i * n_d + j
                level_t = draws[i, j, -1].item()
                for h in range(n_fcst):
                    level_t += rng.normal(0, σ_l[i, j].item())
                    seas = s_full[i, j, (n_train + h) % m].item()
                    noise = rng.normal(0, σ_o[i, j].item())
                    fcst_samples[idx, h] = level_t + seas + noise

        # Rescale to original space
        fcst_samples = fcst_samples * self._y_std + self._y_mean

        if self.clip_negative:
            fcst_samples = np.clip(fcst_samples, 0, None)

        return fcst_samples.mean(axis=0)

    # ── API ────────────────────────────────────────────────────

    def predict_sequence(self, y_test, X_test=None,
                         update_state=True, **kwargs):
        """Predict on test sequence."""
        n_test = len(y_test)
        if _HAS_PYMC and self._trace is not None:
            pred = self._project_forward(n_test)
        else:
            pred = self._predict_mle(n_test)

        if update_state:
            y = np.concatenate(
                [self._y_train, np.asarray(y_test, dtype=float).ravel()])
            self._y_train = y
            self._n_train = len(y)
            self.fit_sequence(y)
        return pred

    def forecast(self, n_periods: int, X_future=None):
        """Forecast future values. Returns (n_periods, 1, 1)."""
        if _HAS_PYMC and self._trace is not None:
            pred = self._project_forward(n_periods)
        else:
            pred = self._predict_mle(n_periods)
        return pred.reshape(-1, 1, 1)

    def _predict_mle(self, n):
        fcst = self._ss_result.get_forecast(n)
        pred = np.asarray(fcst.predicted_mean, dtype=float).ravel()
        if self.clip_negative:
            pred = np.clip(pred, 0, None)
        return pred

    def predict_with_ci(self, n: int, alpha: float = 0.05):
        """Return (mean, lower, upper) credible interval.

        Uses full posterior predictive distribution when PyMC is
        available (matching R's ``$interval`` output).
        """
        if _HAS_PYMC and self._trace is not None:
            # Re-run projection saving full samples
            tr = self._trace
            draws = tr.posterior["level"].values
            σ_l = tr.posterior["sigma_level"].values
            σ_o = tr.posterior["sigma_obs"].values
            s = tr.posterior["s"].values

            n_ch, n_d, _ = draws.shape
            s_full = np.concatenate(
                [s, -s.sum(axis=-1, keepdims=True)], axis=-1)
            m, n_train = self.seasonal_periods, self._n_train

            rng = np.random.default_rng(42)
            fcst = np.zeros((n_ch * n_d, n))
            for i in range(n_ch):
                for j in range(n_d):
                    idx = i * n_d + j
                    lv = draws[i, j, -1].item()
                    for h in range(n):
                        lv += rng.normal(0, σ_l[i, j].item())
                        fcst[idx, h] = lv + s_full[i, j,
                            (n_train + h) % m].item() + \
                            rng.normal(0, σ_o[i, j].item())

            fcst = fcst * self._y_std + self._y_mean
            if self.clip_negative:
                fcst = np.clip(fcst, 0, None)

            mean = fcst.mean(axis=0)
            lower = np.percentile(fcst, alpha / 2 * 100, axis=0)
            upper = np.percentile(fcst, (1 - alpha / 2) * 100, axis=0)
            return mean, lower, upper

        # MLE fallback
        fcst = self._ss_result.get_forecast(n)
        pred = np.asarray(fcst.predicted_mean, dtype=float).ravel()
        ci = np.asarray(fcst.conf_int(alpha=alpha), dtype=float)
        lower, upper = ci[:, 0], ci[:, 1]
        if self.clip_negative:
            pred = np.clip(pred, 0, None)
            lower = np.clip(lower, 0, None)
            upper = np.clip(upper, 0, None)
        return pred, lower, upper

    def __repr__(self) -> str:
        return (
            f"BSTSForecaster(period={self.seasonal_periods}, "
            f"niter={self.niter}, burn={self.burn})"
        )
