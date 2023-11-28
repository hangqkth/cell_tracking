import numpy as np
import matplotlib.pyplot as plt

# Known parameters of the simulation
deltaTime = 0.1     # Time between samples
N = 100             # Length of the simulation

# Position matrix
A = np.eye(2)

# Observation matrix
C = np.eye(2)

# Covariance noise in the positions
Q = 1e-3 * np.eye(2)

# Covariance noise in the observations
R = 0.001 * np.eye(2)

# Initial values
xHat = np.zeros((2, 1))
covHat = np.zeros((2, 2))

# Initial position
x = np.zeros((2, 1))

# Arrays to store states, observations, and estimates
states = np.zeros((2, N))
obs = np.zeros((2, N))
est = np.zeros((2, N))

for n in range(N):
    # State update
    x = A @ x + np.random.multivariate_normal(np.zeros(2), Q).reshape(2, 1)

    # Observation
    y = C @ x + np.random.multivariate_normal(np.zeros(2), R).reshape(2, 1)

    # Store history
    states[:, n] = x.flatten()
    obs[:, n] = y.flatten()

    # Predicted state and covariance
    xPred = A @ xHat
    covPred = A @ covHat @ A.T + Q

    # Kalman gain
    S = C @ covPred @ C.T + R
    B = C @ covPred
    klm_gain = np.linalg.inv(S) @ B

    # Update estimates
    xHat = xPred + klm_gain @ (y - C @ xPred)
    covHat = covPred - klm_gain @ C @ covPred
    est[:, n] = xHat.flatten()

# Plotting
plt.figure()
plt.plot(states[0, :], 'o', label='True States')
plt.plot(obs[0, :], 'rx', label='Observations')
plt.plot(est[0, :], 'k-', label='Estimate')
# plt.legend(loc='northeast')
plt.xlabel('Sample')
plt.ylabel('Position X')
plt.title('Estimate positions in x-axis')
plt.grid(True)

plt.figure()
plt.plot(states[1, :], 'o', label='True States')
plt.plot(obs[1, :], 'rx', label='Observations')
plt.plot(est[1, :], 'k-', label='Estimate')
# plt.legend(loc='northeast')
plt.xlabel('Sample')
plt.ylabel('Position Y')
plt.title('Estimate positions in y-axis')
plt.grid(True)

plt.show()
