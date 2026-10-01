"""Reproduce the convergence studies and figures of the report.

Usage (from part2_participating_annuity/):
    python scripts/run_experiments.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pricing import Params, analytical_value, binomial_tree_value, monte_carlo_value  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
p = Params()

# --- Reference value -------------------------------------------------------
gmab, gmdb, v0 = analytical_value(p)
print(f"Analytical  GMAB = {gmab:,.4f} | GMDB = {gmdb:,.4f} | Total = {v0:,.4f}")

# --- Binomial tree convergence ---------------------------------------------
steps = list(range(400, 10_001, 400))
tree = [binomial_tree_value(p, N=n) for n in steps]
print(f"Binomial tree (N=10,000) = {tree[-1]:,.4f}")

plt.figure(figsize=(10, 6))
plt.plot(steps, tree, marker="o", lw=2, label="Binomial tree value")
plt.axhline(v0, ls="--", lw=2, color="C1", label=f"Reference value = {v0:,.2f}")
plt.xlabel("Number of steps (N)"); plt.ylabel("Contract value (€)")
plt.title("Convergence of binomial tree pricing")
plt.grid(True, ls="--", alpha=0.6); plt.legend()
plt.gca().ticklabel_format(style="plain", axis="y", useOffset=False)
plt.tight_layout(); plt.savefig(FIG / "binomial_tree_convergence.png", dpi=200); plt.close()

# --- Monte-Carlo convergence -----------------------------------------------
Ms = [100, 500, 1000, 2000, 5000, 10_000]
K = 20  # repetitions per M, to estimate the standard deviation
means, stds = [], []
for M in Ms:
    est = [monte_carlo_value(p, N=1000, M=M, seed=1000 * M + k) for k in range(K)]
    means.append(np.mean(est)); stds.append(np.std(est))
    print(f"MC M={M:>6}: mean = {means[-1]:,.2f}  std = {stds[-1]:,.2f}")

plt.figure(figsize=(10, 6))
plt.errorbar(Ms, means, yerr=stds, fmt="o-", capsize=5, label="Monte-Carlo (mean ± std)")
plt.axhline(v0, ls="--", lw=2, color="C1", label=f"Reference value = {v0:,.2f}")
plt.xlabel("Number of simulations (M)"); plt.ylabel("Contract value (€)")
plt.title("Convergence of Monte-Carlo pricing"); plt.grid(True, alpha=0.6); plt.legend()
plt.tight_layout(); plt.savefig(FIG / "monte_carlo_convergence.png", dpi=200); plt.close()
