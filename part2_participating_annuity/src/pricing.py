"""Pricing of a participating annuity with a guaranteed minimum benefit.

Contract (see docs/report_part2_participating_annuity.pdf):
  * Survival at T   -> C0 * max(S_T / S_0, exp(g T))      (GMAB)
  * Death at t <= T -> C0 * max(S_t / S_0, exp(g t))      (GMDB)

Market: Black-Scholes (GBM, constant risk-free rate r, volatility sigma).
Mortality: constant force of mortality mu, independent of the market.

Three independent methods are implemented and cross-checked:
  1. ``analytical_value``    - closed form (Black-Scholes) + numerical integration
  2. ``binomial_tree_value`` - recombining binomial tree with mortality
  3. ``monte_carlo_value``   - Euler-discretised paths, mortality by weighting
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class Params:
    S0: float = 100.0      # initial index level
    C0: float = 100_000.0  # initial investment
    r: float = 0.032       # continuous risk-free rate
    sigma: float = 0.12    # index volatility
    g: float = 0.025       # guaranteed (continuous) rate
    mu: float = 0.02       # constant force of mortality
    T: float = 5.0         # maturity (years)


def _bs_call(S0, K, T, r, sigma):
    """Black-Scholes price of a European call (vectorised in K and T)."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _guaranteed_leg_value(t, p: Params):
    """Value at 0 of C0*max(S_t/S0, e^{g t}) paid at t (no mortality)."""
    K = p.S0 * np.exp(p.g * t)
    return p.C0 * np.exp((p.g - p.r) * t) + (p.C0 / p.S0) * _bs_call(p.S0, K, t, p.r, p.sigma)


# --------------------------------------------------------------------------- #
# 1. Analytical value
# --------------------------------------------------------------------------- #
def analytical_value(p: Params = Params(), n_steps: int = 1_000_000):
    """Return (GMAB, GMDB, total). The GMDB integral is a Riemann sum over n_steps."""
    survival = lambda t: np.exp(-p.mu * t)

    gmab = survival(p.T) * _guaranteed_leg_value(p.T, p)

    dt = p.T / n_steps
    t = np.arange(1, n_steps + 1) * dt
    death_prob = survival(t - dt) - survival(t)  # P(t-dt < tau <= t)
    gmdb = float(np.sum(death_prob * _guaranteed_leg_value(t, p)))

    return float(gmab), gmdb, float(gmab) + gmdb


# --------------------------------------------------------------------------- #
# 2. Binomial tree
# --------------------------------------------------------------------------- #
def binomial_tree_value(p: Params = Params(), N: int = 1000) -> float:
    """Backward induction on a recombining tree (O(N) memory).

    At each node the value is a probability-weighted mix of
    (i) the discounted risk-neutral continuation value (survival over dt) and
    (ii) the death benefit paid immediately (death over dt).
    """
    dt = p.T / N
    drift = (p.r - 0.5 * p.sigma**2) * dt
    u = np.exp(drift + p.sigma * np.sqrt(dt))
    d = np.exp(drift - p.sigma * np.sqrt(dt))
    q = (np.exp(p.r * dt) - d) / (u - d)  # risk-neutral up probability

    surv = np.exp(-p.mu * dt)
    disc = np.exp(-p.r * dt)

    # Terminal payoff (j = number of up moves)
    j = np.arange(N + 1)
    S_T = p.S0 * u**j * d ** (N - j)
    V = p.C0 * np.maximum(S_T / p.S0, np.exp(p.g * p.T))

    for i in range(N - 1, -1, -1):
        t = i * dt
        j = np.arange(i + 1)
        S_t = p.S0 * u**j * d ** (i - j)
        continuation = disc * (q * V[1 : i + 2] + (1 - q) * V[: i + 1])
        death_payoff = p.C0 * np.maximum(S_t / p.S0, np.exp(p.g * t))
        V = surv * continuation + (1 - surv) * death_payoff

    return float(V[0])


# --------------------------------------------------------------------------- #
# 3. Monte-Carlo
# --------------------------------------------------------------------------- #
def monte_carlo_value(p: Params = Params(), N: int = 1000, M: int = 1000, seed=None) -> float:
    """Simulate M index paths (Euler scheme, N steps); weight discounted cash
    flows by death/survival probabilities instead of simulating mortality."""
    rng = np.random.default_rng(seed)
    dt = p.T / N
    t = np.arange(1, N + 1) * dt

    eps = rng.standard_normal((M, N))
    S = p.S0 * np.cumprod(1 + p.r * dt + p.sigma * np.sqrt(dt) * eps, axis=1)

    discounted_payoff = p.C0 * np.maximum(S / p.S0, np.exp(p.g * t)) * np.exp(-p.r * t)

    survival = lambda s: np.exp(-p.mu * s)
    weights = survival(t - dt) - survival(t)  # P(t_{i-1} < tau <= t_i)
    weights[-1] += survival(p.T)              # survivors are paid at T

    return float(np.mean(discounted_payoff @ weights))
