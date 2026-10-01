import numpy as np

def annuity_binomial_tree(S0=100, C0=100000, r=0.032, sigma=0.12, g=0.025, mu=0.02,T=5, N=100):
    dt = T / N

    u = np.exp((r - sigma**2 / 2) * dt + sigma * np.sqrt(dt))
    d = np.exp((r - sigma**2 / 2) * dt - sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)

    # Stock price at each node
    stock = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(i+1):
            stock[j, i] = S0 * (u**j) * (d**(i-j))

    # Value of the contract at each node
    value = np.zeros((N+1, N+1))

    # Payoff at maturity
    for j in range(N+1):
        ST = stock[j, N]
        payoff = max(ST / S0, np.exp(g*T))
        value[j, N] = C0 * payoff

    # Backward induction
    for i in range(N-1, -1, -1):
        t = i * dt

        for j in range(i+1):
            St = stock[j, i]

            # Continuation payoff (obtained if survives during [t, t+dt])
            continuation = np.exp(-r * dt) * (p * value[j+1, i+1] + (1-p) * value[j, i+1])

            # Death payoff (obtained if dies during [t, t+dt])
            death_payoff = C0 * max(St / S0, np.exp(g * t))
            
            # Probability of survival over dt
            survival_prob = np.exp(-mu * dt)

            # Probability of death over dt
            death_prob = 1 - np.exp(-mu * dt)

            # Expected value
            value[j, i] = (survival_prob * continuation + death_prob * death_payoff)

    return value[0, 0]

if __name__ == "__main__":
    N = 1000
    contract_value = annuity_binomial_tree(N=N)
    print(f"Steps: {N} - Contract value: {contract_value}")