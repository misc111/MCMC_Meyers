from __future__ import annotations

import math
import random
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import chainladder as cl
import numpy as np


# --------------------------
# Data structures
# --------------------------


@dataclass
class LCLData:
    W: int
    D: int
    C: np.ndarray  # cumulative triangle, shape (W, D)
    obs_mask: np.ndarray  # boolean mask, True where observed
    premiums: np.ndarray  # shape (W,)


def triangle_to_lcl_data(
    triangle: cl.Triangle,
    column: Optional[str] = None,
    premiums: Optional[np.ndarray] = None,
) -> LCLData:
    """Convert a chainladder Triangle into the internal LCLData structure."""

    tri = triangle
    if tri.is_val_tri:
        tri = tri.val_to_dev()
    if not tri.is_cumulative:
        tri = tri.incr_to_cum()

    if column is not None:
        tri = tri.loc[:, column]

    index_dim, column_dim, origin_dim, dev_dim = tri.shape
    if index_dim != 1:
        raise ValueError(
            f"triangle_to_lcl_data expects a single index dimension, got {index_dim}."
        )
    if column_dim != 1:
        raise ValueError(
            f"triangle_to_lcl_data expects a single value column, got {column_dim}."
        )

    # Extract the 2D cumulative triangle
    C = np.asarray(tri.values[0, 0], dtype=float)
    obs_mask = np.isfinite(C)

    W, D = C.shape

    if premiums is None:
        latest = tri.latest_diagonal.values
        premiums = latest.reshape(-1).astype(float)
    else:
        premiums = np.asarray(premiums, dtype=float)

    premiums = np.nan_to_num(premiums, nan=1.0, posinf=1.0, neginf=1.0)
    if premiums.shape[0] != W:
        raise ValueError(
            f"Premium vector length {premiums.shape[0]} does not match origin count {W}."
        )
    premiums = np.maximum(1e-3, premiums)

    return LCLData(W=W, D=D, C=C, obs_mask=obs_mask, premiums=premiums)


def build_sample_dataset() -> LCLData:
    # Use chainladder's built-in RAA dataset as the sample triangle
    sample_tri = cl.load_sample("raa")
    # Smooth premiums to a reasonable scale for the Bayesian prior
    latest = sample_tri.latest_diagonal.values.reshape(-1)
    premiums = np.maximum(1.0, latest / 10.0)
    return triangle_to_lcl_data(sample_tri, premiums=premiums)


# --------------------------
# Utilities
# --------------------------


def reflect(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    # Reflect values into [lo, hi] interval for symmetric RW proposals
    span = hi - lo
    y = x.copy()
    if span <= 0:
        return np.clip(y, lo, hi)
    # Bring into infinite tiling then reflect odd tiles
    y = (y - lo) % (2 * span)
    over = y > span
    y[over] = 2 * span - y[over]
    return y + lo


def to_full_beta(beta_free: np.ndarray, D: int) -> np.ndarray:
    b = np.zeros(D)
    if D > 1:
        b[: D - 1] = beta_free
    b[D - 1] = 0.0
    return b


def a_to_sigma(a: np.ndarray) -> np.ndarray:
    # sigma_d = sum_{i=d}^{D} a_i
    # a shape (D,)
    return np.array([np.sum(a[d:]) for d in range(len(a))])


def ols_init(data: LCLData) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    OLS initialization for alpha, beta (with beta_D fixed to 0), sigma, logELR.
    """
    W, D = data.W, data.D
    obs = data.obs_mask
    y = np.log(np.where(obs, data.C, np.nan))
    rows: List[Tuple[int, int, float]] = []
    for w in range(W):
        for d in range(D):
            if obs[w, d] and np.isfinite(y[w, d]):
                rows.append((w, d, float(y[w, d])))

    n = len(rows)
    p = W + (D - 1)
    X = np.zeros((n, p))
    Y = np.zeros(n)
    for i, (w, d, val) in enumerate(rows):
        # alpha_w
        X[i, w] = 1.0
        # beta_d for d < D (free params); beta_D fixed to 0
        if d < D - 1:
            X[i, W + d] = 1.0
        Y[i] = val

    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    alpha0 = coef[:W]
    beta_free0 = coef[W:]
    beta0 = to_full_beta(beta_free0, D)

    # Residuals by dev to seed sigma
    mu_hat = np.zeros_like(y)
    for w in range(W):
        for d in range(D):
            mu_hat[w, d] = alpha0[w] + beta0[d]
    resid = []
    for d in range(D):
        r = []
        for w in range(W):
            if obs[w, d] and np.isfinite(y[w, d]):
                r.append(y[w, d] - mu_hat[w, d])
        if len(r) == 0:
            resid.append(0.2)
        else:
            resid.append(float(np.sqrt(np.maximum(1e-6, np.var(r)))))
    # Enforce monotone decreasing
    sigma0 = np.maximum(0.05, np.minimum(0.6, np.array(resid)))
    for d in range(1, D):
        sigma0[d] = min(sigma0[d], sigma0[d - 1] - 1e-3)
    sigma0 = np.maximum(sigma0, 0.05)
    a0 = np.zeros(D)
    for d in range(D - 1):
        a0[d] = max(0.001, sigma0[d] - sigma0[d + 1])
    a0[D - 1] = max(0.001, sigma0[D - 1])
    # clamp into (0,1)
    a0 = np.clip(a0, 0.001, 0.9)

    # crude logELR init by centering alpha around log premium
    logELR0 = float(np.mean(alpha0 - np.log(data.premiums)))
    logELR0 = float(np.clip(logELR0, -1.0, 0.5))

    return alpha0, beta_free0, a0, logELR0


# --------------------------
# MCMC
# --------------------------


def log_normal_pdf(x: float, mean: float, sd: float) -> float:
    var = sd * sd
    return -0.5 * math.log(2 * math.pi * var) - 0.5 * ((x - mean) ** 2) / var


class LCLPosteriorSamples:
    def __init__(self, W: int, D: int):
        self.W, self.D = W, D
        self.alpha: List[np.ndarray] = []
        self.beta_free: List[np.ndarray] = []
        self.a: List[np.ndarray] = []
        self.logELR: List[float] = []

    @property
    def num_draws(self) -> int:
        return len(self.logELR)

    def get_draw(self, i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        return self.alpha[i], self.beta_free[i], self.a[i], self.logELR[i]


class LCLMCMCRunner:
    def __init__(
        self,
        data: LCLData,
        iterations: int,
        burn_in: int,
        thin: int,
        adapt: int,
        chains: int,
        update_every: int,
        trace_param: str = "logELR",
        tau_overrides: Optional[Dict[str, float]] = None,
        on_update: Optional[Callable[[Dict], None]] = None,
    ):
        self.data = data
        self.iterations = iterations
        self.burn_in = burn_in
        self.thin = max(1, thin)
        self.adapt = max(0, adapt)
        self.chains = max(1, chains)
        self.update_every = max(1, update_every)
        self.trace_param = trace_param
        self.tau_overrides = tau_overrides or {}
        self.on_update = on_update
        self._stop = threading.Event()

        # Prior hyperparams
        self.alpha_sd = 10 ** (-0.5)  # ~0.316

        # Default RW scales
        self.tau_alpha = float(self.tau_overrides.get("alpha", 0.1))
        self.tau_beta = float(self.tau_overrides.get("beta", 0.05))
        self.tau_a = float(self.tau_overrides.get("a", 0.05))
        self.tau_logELR = float(self.tau_overrides.get("logELR", 0.05))

    def request_stop(self):
        self._stop.set()

    # ----- Likelihood and prior -----
    def log_likelihood(self, alpha: np.ndarray, beta_free: np.ndarray, a: np.ndarray) -> float:
        W, D = self.data.W, self.data.D
        beta = to_full_beta(beta_free, D)
        sigma = a_to_sigma(a)
        y = np.log(np.where(self.data.obs_mask, self.data.C, np.nan))
        ll = 0.0
        for w in range(W):
            for d in range(D):
                if self.data.obs_mask[w, d] and np.isfinite(y[w, d]):
                    mu = alpha[w] + beta[d]
                    sd = sigma[d]
                    sd = max(1e-6, float(sd))
                    ll += log_normal_pdf(float(y[w, d]), float(mu), sd)
        return float(ll)

    def log_prior(self, alpha: np.ndarray, beta_free: np.ndarray, a: np.ndarray, logELR: float) -> float:
        # Supports: beta in [-5,5], a in [0,1], logELR in [-1,0.5]
        if not (-1.0 <= logELR <= 0.5):
            return -math.inf
        if np.any(beta_free < -5.0) or np.any(beta_free > 5.0):
            return -math.inf
        if np.any(a < 0.0) or np.any(a > 1.0):
            return -math.inf
        # alpha prior: N(log(premium_w) + logELR, sd)
        mean_alpha = np.log(self.data.premiums) + logELR
        var = self.alpha_sd ** 2
        const = -0.5 * math.log(2 * math.pi * var)
        lp_alpha = float(np.sum(const - 0.5 * ((alpha - mean_alpha) ** 2) / var))
        # Light prior on a ~ Beta(2,2) to discourage extremes and sigma collapse
        eps = 1e-12
        a_safe = np.clip(a, eps, 1.0 - eps)
        lp_a = float(np.sum((2.0 - 1.0) * np.log(a_safe) + (2.0 - 1.0) * np.log(1.0 - a_safe)))
        return lp_alpha + lp_a

    def log_posterior(self, alpha: np.ndarray, beta_free: np.ndarray, a: np.ndarray, logELR: float) -> float:
        lp = self.log_prior(alpha, beta_free, a, logELR)
        if not np.isfinite(lp):
            return -math.inf
        ll = self.log_likelihood(alpha, beta_free, a)
        return lp + ll

    # ----- MCMC steps -----
    def run_chain(self, chain_id: int) -> LCLPosteriorSamples:
        W, D = self.data.W, self.data.D
        alpha, beta_free, a, logELR = ols_init(self.data)

        # Adapt stats
        acc_alpha = 0
        acc_beta = 0
        acc_a = 0
        acc_logELR = 0
        tries_alpha = 0
        tries_beta = 0
        tries_a = 0
        tries_logELR = 0

        # Current posterior
        cur_lp = self.log_posterior(alpha, beta_free, a, logELR)

        samples = LCLPosteriorSamples(W, D)

        for t in range(1, self.iterations + 1):
            if self._stop.is_set():
                break

            # Block A: alpha (component-wise scalar updates)
            for w in range(W):
                tries_alpha += 1
                prop_alpha = alpha.copy()
                prop_alpha[w] = alpha[w] + np.random.normal(0.0, self.tau_alpha)
                prop_lp = self.log_posterior(prop_alpha, beta_free, a, logELR)
                if math.isfinite(prop_lp) and (math.log(random.random()) < (prop_lp - cur_lp)):
                    alpha = prop_alpha
                    cur_lp = prop_lp
                    acc_alpha += 1

            # Block B: beta_free in [-5,5]
            prop_beta = beta_free + np.random.normal(0.0, self.tau_beta, size=D - 1)
            prop_beta = reflect(prop_beta, -5.0, 5.0)
            tries_beta += 1
            prop_lp = self.log_posterior(alpha, prop_beta, a, logELR)
            if math.isfinite(prop_lp) and (math.log(random.random()) < (prop_lp - cur_lp)):
                beta_free = prop_beta
                cur_lp = prop_lp
                acc_beta += 1

            # Block C: a in [0,1]
            prop_a = a + np.random.normal(0.0, self.tau_a, size=D)
            prop_a = reflect(prop_a, 0.0, 1.0)
            tries_a += 1
            prop_lp = self.log_posterior(alpha, beta_free, prop_a, logELR)
            if math.isfinite(prop_lp) and (math.log(random.random()) < (prop_lp - cur_lp)):
                a = prop_a
                cur_lp = prop_lp
                acc_a += 1

            # Block D: logELR in [-1,0.5]
            prop_logELR = float(reflect(np.array([logELR + np.random.normal(0.0, self.tau_logELR)]), -1.0, 0.5)[0])
            tries_logELR += 1
            prop_lp = self.log_posterior(alpha, beta_free, a, prop_logELR)
            if math.isfinite(prop_lp) and (math.log(random.random()) < (prop_lp - cur_lp)):
                logELR = prop_logELR
                cur_lp = prop_lp
                acc_logELR += 1

            # Adaptation phase
            if t <= self.adapt and (t % self.update_every == 0):
                # Target acceptance 0.25-0.40 vectors, ~0.5 scalar
                def tune(scale: float, acc: int, tries: int, target_lo: float, target_hi: float) -> float:
                    if tries == 0:
                        return scale
                    rate = acc / max(1, tries)
                    if rate < target_lo:
                        scale *= 0.9
                    elif rate > target_hi:
                        scale *= 1.1
                    return float(max(1e-4, min(2.0, scale)))

                # Alpha uses scalar updates; target higher acceptance
                self.tau_alpha = tune(self.tau_alpha, acc_alpha, tries_alpha, 0.45, 0.6)
                self.tau_beta = tune(self.tau_beta, acc_beta, tries_beta, 0.25, 0.4)
                self.tau_a = tune(self.tau_a, acc_a, tries_a, 0.25, 0.4)
                self.tau_logELR = tune(self.tau_logELR, acc_logELR, tries_logELR, 0.45, 0.6)
                acc_alpha = acc_beta = acc_a = acc_logELR = 0
                tries_alpha = tries_beta = tries_a = tries_logELR = 0

            # Save samples after burn-in and thinning
            if t > self.burn_in and ((t - self.burn_in) % self.thin == 0):
                samples.alpha.append(alpha.copy())
                samples.beta_free.append(beta_free.copy())
                samples.a.append(a.copy())
                samples.logELR.append(float(logELR))

            # Stream progress (full bundle: logELR, alpha[], beta[], sigma[])
            if self.on_update and (t % self.update_every == 0):
                beta_full = to_full_beta(beta_free, D)
                sigma = a_to_sigma(a)
                info = dict(
                    type="progress",
                    chain=chain_id,
                    iter=t,
                    tau=dict(alpha=self.tau_alpha, beta=self.tau_beta, a=self.tau_a, logELR=self.tau_logELR),
                    traceBundle=dict(
                        logELR=float(logELR),
                        alpha=[float(v) for v in alpha.tolist()],
                        beta=[float(v) for v in beta_full.tolist()],
                        sigma=[float(v) for v in sigma.tolist()],
                    ),
                )
                try:
                    self.on_update(info)
                except Exception:
                    pass

        return samples

    def run(self) -> LCLPosteriorSamples:
        all_samples: Optional[LCLPosteriorSamples] = None
        for c in range(self.chains):
            if self._stop.is_set():
                break
            if self.on_update:
                self.on_update({"type": "chain_start", "chain": c + 1})
            samples = self.run_chain(chain_id=c + 1)
            if all_samples is None:
                all_samples = samples
            else:
                all_samples.alpha.extend(samples.alpha)
                all_samples.beta_free.extend(samples.beta_free)
                all_samples.a.extend(samples.a)
                all_samples.logELR.extend(samples.logELR)
            if self.on_update:
                self.on_update({"type": "chain_done", "chain": c + 1, "draws": samples.num_draws})
        if self.on_update:
            self.on_update({"type": "done"})
        return all_samples or LCLPosteriorSamples(self.data.W, self.data.D)


# --------------------------
# Posterior predictive simulation
# --------------------------


def simulate_triangles_from_samples(
    data: LCLData, samples: LCLPosteriorSamples, indices: List[int]
) -> Dict[str, object]:
    W, D = data.W, data.D
    rng = np.random.default_rng(123)
    triangles: Dict[str, List[List[float]]] = {}
    totals_ultimate: List[float] = []
    totals_reserve: List[float] = []

    # Latest observed cumulative per AY (last observed dev based on mask)
    latest_obs = np.zeros(W)
    for w in range(W):
        observed_devs = np.where(data.obs_mask[w])[0]
        if observed_devs.size > 0:
            last_d = int(observed_devs.max())
            val = data.C[w, last_d]
            latest_obs[w] = float(val) if np.isfinite(val) else 0.0
        else:
            latest_obs[w] = 0.0

    for idx in indices:
        alpha, beta_free, a, _ = samples.get_draw(idx)
        beta = to_full_beta(beta_free, D)
        sigma = a_to_sigma(a)
        tri = np.array(data.C)  # copy, includes NaNs for missing
        for w in range(W):
            for d in range(D):
                if not data.obs_mask[w, d]:
                    mu = alpha[w] + beta[d]
                    sd = max(1e-6, float(sigma[d]))
                    tri[w, d] = float(rng.lognormal(mean=float(mu), sigma=sd))

        # Ultimate per AY is C[w, D-1]
        ult = np.nan_to_num(tri[:, D - 1], nan=0.0)
        total_ult = float(np.sum(ult))
        # Reserve = total ultimate - sum(latest observed)
        total_res = float(np.sum(np.maximum(0.0, ult - latest_obs)))
        totals_ultimate.append(total_ult)
        totals_reserve.append(total_res)
        triangles[str(idx)] = tri.tolist()

    # Build histogram (simple fixed binning)
    def hist(vals: List[float], bins: int = 40) -> Dict[str, object]:
        if not vals:
            return {"bins": [], "counts": []}
        v = np.array(vals)
        lo, hi = float(np.min(v)), float(np.max(v))
        if hi == lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, bins + 1)
        counts, _ = np.histogram(v, bins=edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return {"bins": centers.tolist(), "counts": counts.astype(int).tolist()}

    def summary(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {
                "mean": 0.0,
                "p05": 0.0,
                "p50": 0.0,
                "p95": 0.0,
            }
        v = np.array(vals)
        return {
            "mean": float(np.mean(v)),
            "p05": float(np.quantile(v, 0.05)),
            "p50": float(np.quantile(v, 0.50)),
            "p95": float(np.quantile(v, 0.95)),
        }

    return {
        "triangles": triangles,
        "hist": {
            "ultimate": hist(totals_ultimate),
            "reserve": hist(totals_reserve),
        },
        "summary": {
            "ultimate": summary(totals_ultimate),
            "reserve": summary(totals_reserve),
        },
    }
