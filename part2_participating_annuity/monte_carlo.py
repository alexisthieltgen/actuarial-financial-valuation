import numpy as np
import matplotlib.pyplot as plt

def monte_carlo(S0=100, C0=100000, r=0.032, sigma=0.12, g=0.025, mu=0.02,T=5, N=100, M=50):

	def survival_prob(t):
		return np.exp(-mu * t) # P(tau > t)

	def death_prob(t, dt):
		return survival_prob(t - dt) - survival_prob(t) # P(t - dt < tau <= t) = P(tau > t - dt) - P(tau > t)
	
	dt = T / N

	payoffs = np.zeros((M,N))

	for m in range(M):
		#print(f"Simulation {m+1}/{M}")
		S_path = np.zeros(N+1)
		St = S0
		S_path[0] = S0
		for i in range(1, N+1):
			eps = np.random.normal()
			delta_S = St*(r*dt + sigma*np.sqrt(dt)*eps)
			St = St + delta_S
			S_path[i] = St

		for i in range(N):
			t = (i+1)*dt
			St = S_path[i+1]
			payoffs[m,i] = C0*max(St / S0, np.exp(g*t))*np.exp(-r*t)

	#plt.plot(np.linspace(0,5,N+1),S_path)
	#plt.show()

	probs = np.zeros(N)
	for i in range(N):
		t = (i+1)*dt
		probs[i] = death_prob(t, dt)
	probs[N-1] += survival_prob(T)

	sim_values = payoffs @ probs

	contract_value = np.mean(sim_values)

	return contract_value

if __name__ == "__main__":
	contract_value = monte_carlo(N=10000, M=1000)
	print(contract_value)