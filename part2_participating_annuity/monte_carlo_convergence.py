import numpy as np
import matplotlib.pyplot as plt
from monte_carlo import monte_carlo
from numerical_value import compute_numerical_value

# Parameters
S0 = 100
C0 = 100000
r = 0.032
sigma = 0.12
g = 0.025
mu = 0.02
T = 5

# Number of repetitions per M (to compute std)
K = 20 

simulations = [100, 500, 1000, 2000, 5000, 10000]
means = []
stds = []

for M in simulations:
    print(f"Running Monte Carlo with M={M} simulations...")
    estimates = []
    
    for k in range(K):
        print(f"  Repetition {k+1}/{K}")
        val = monte_carlo(S0=S0, C0=C0, r=r, sigma=sigma, g=g, mu=mu, T=T, N=1000, M=M)
        estimates.append(val)
    
    means.append(np.mean(estimates))
    stds.append(np.std(estimates))

GMAB, GMDB, V0 = compute_numerical_value(S0=S0, C0=C0, r=r, sigma=sigma, g=g, mu=mu, T=T)

plt.figure(figsize=(10, 6))
plt.errorbar(simulations, means, yerr=stds, fmt='o-', capsize=5)
plt.axhline(y=V0, linestyle='--', linewidth=2, label=f'Numerical value = {V0:.4f}')
for x, y in zip(simulations, means):
    plt.text(x, y, f'{y:.0f}', ha='center', va='bottom')
plt.xlabel("Number of simulations (M)")
plt.ylabel("Contract value (€)")
plt.title("Convergence of Monte-Carlo pricing (mean ± std)")
plt.grid()
plt.legend()
plt.savefig("monte_carlo_convergence.pdf", dpi=300)
plt.show()