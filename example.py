import numpy as np

def compute_P_ej_k(y, ej_size, k):
    n = ej_size
    C = np.exp(2j * np.pi / (n + 1))
    result = 0
    for l in range(n + 1):
        term = C ** (-l * k)
        prod = 1
        for m in range(1, n + 1):
            prod *= (1 + (C ** l - 1) * y[m - 1])
        result += term * prod
    return result.real / (n + 1)

def dy_dt(y, delta, lambda_, hyperedges, lambda_star, thresholds):
    dydt = np.zeros_like(y)
    for i in range(len(y)):
        sum_term = 0
        for ej in hyperedges:
            if i in ej:
                ej_size = len(ej)
                lambda_star_ej = lambda_star(ej_size)
                for k in range(thresholds[ej], ej_size + 1):
                    P_ej_k = compute_P_ej_k([y[node] for node in ej], ej_size, k)
                    sum_term += lambda_star_ej * P_ej_k
        dydt[i] = -delta * y[i] + lambda_ * (1 - y[i]) * sum_term
    return dydt

def runge_kutta_step(y, dt, delta, lambda_, hyperedges, lambda_star, thresholds):
    k1 = dy_dt(y, delta, lambda_, hyperedges, lambda_star, thresholds)
    k2 = dy_dt(y + dt/2 * k1, delta, lambda_, hyperedges, lambda_star, thresholds)
    k3 = dy_dt(y + dt/2 * k2, delta, lambda_, hyperedges, lambda_star, thresholds)
    k4 = dy_dt(y + dt * k3, delta, lambda_, hyperedges, lambda_star, thresholds)
    return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

# Example usage with a larger toy hypergraph
y = np.array([1, 0., 0., 1., 0., 1.])  # Initial probabilities of being active
delta = 0.5
lambda_ = 0.25
hyperedges = [(0, 1, 2), (2, 3, 4), (0, 5, 1)]  # Use tuples instead of lists
lambda_star = lambda size: np.log2(size) # Lambda star function
thresholds = {(0, 1, 2): 2, (1, 2, 3): 2, (2, 3, 4): 2, (0, 3, 4): 2, (0, 1, 4): 2, (0, 5, 1): 2}  # Thresholds for activation

# Runge-Kutta parameters
dt = 0.01  # Time step
num_steps = 1000  # Number of integration steps

# make a funciton of  the average
average_func = []

# Integrate over time
for step in range(num_steps):
    average_func.append(np.mean(y))
    y = runge_kutta_step(y, dt, delta, lambda_, hyperedges, lambda_star, thresholds)

print(average_func)


