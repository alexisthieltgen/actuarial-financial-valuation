from binomial_tree import annuity_binomial_tree
from numerical_value import compute_numerical_value
import matplotlib.pyplot as plt

# Parameters
S0 = 100
C0 = 100000
r = 0.032
sigma = 0.12
g = 0.025
mu = 0.02
T = 5

steps = range(400, 10001, 400)
prices = []

for N in steps:
    price = annuity_binomial_tree(S0=S0, C0=C0, r=r, sigma=sigma, g=g, mu=mu, T=T, N=N)
    prices.append(price)
    print(f"Steps: {N} - Contract value: {price}")
    
GMAB, GMDB, V0 = compute_numerical_value(S0=S0, C0=C0, r=r, sigma=sigma, g=g, mu=mu, T=T)
print(f"Contract value: {V0:.4f}, GMAB: {GMAB:.4f}, GMDB: {GMDB:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(steps, prices, marker='o', linewidth=2, label="Binomial tree value")
plt.axhline(y=V0, linestyle='--', linewidth=2, label=f'Numerical value = {V0:.4f}')
plt.xlabel("Number of steps (N)", fontsize=12)
plt.ylabel("Contract value (€)", fontsize=12)
plt.title("Convergence of binomial tree pricing", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
ax = plt.gca()
ax.ticklabel_format(style='plain', axis='y', useOffset=False)
plt.savefig("binomial_tree_convergence_2.pdf", dpi=300)
plt.show()
