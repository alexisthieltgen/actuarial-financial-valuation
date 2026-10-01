import numpy as np
from scipy.stats import norm

def compute_numerical_value(S0=100, C0=100000, r=0.032, sigma=0.12, g=0.025, mu=0.02, T=5):

	def survival_prob(t):
		return np.exp(-mu * t) # P(tau > t)

	def death_prob(t, dt):
		return survival_prob(t - dt) - survival_prob(t) # P(t - dt < tau <= t) = P(tau > t - dt) - P(tau > t)

	def bs_call(S0, K, T):
		if T == 0:
			return max(S0 - K, 0)
		
		d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
		d2 = d1 - sigma * np.sqrt(T)
		
		return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

	# GMAB
	K_T = S0 * np.exp(g * T)

	GMAB = survival_prob(T) * (C0 * np.exp((g - r) * T) + (C0 / S0) * bs_call(S0, K_T, T))

	# GMDB
	N = int(1e6) # Number of integration steps
	dt = T / N

	GMDB = 0

	for i in range(1, N+1):
		t = i * dt
		
		K_t = S0 * np.exp(g * t)
		
		term = (C0 * np.exp((g - r) * t) + (C0 / S0) * bs_call(S0, K_t, t))
		
		GMDB += death_prob(t, dt) * term

	# Total value
	V0 = GMAB + GMDB

	return GMAB, GMDB, V0

if __name__ == "__main__":
	GMAB, GMDB, V0 = compute_numerical_value()
	print(f"Contract value: {V0:.4f}, GMAB: {GMAB:.4f}, GMDB: {GMDB:.4f}")