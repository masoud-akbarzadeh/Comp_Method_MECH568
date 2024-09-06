import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 1125  # Speed of sound in water (ft/s)
L = 100   # Length of the pipe (ft)
dx = 1    # Spatial step (ft)
dt = 0.001  # Time step (s)
T = 0.05  # Total time (s)

# Derived constants
nx = int(L / dx) + 1  # Number of spatial points
nt = int(T / dt) + 1  # Number of time steps
r = (c * dt / dx) ** 2

# Initial conditions
P_explicit = np.ones(nx) * 850  # Explicit scheme pressure array
P_implicit = np.ones(nx) * 850  # Implicit scheme pressure array

# Boundary conditions
P_explicit[0] = 0  # Boundary at x=0 is 0 psi
P_implicit[0] = 0  # Boundary at x=0 is 0 psi

# Arrays to store the previous time step
P_explicit_prev = P_explicit.copy()
P_implicit_prev = P_implicit.copy()

# Implicit scheme coefficients
A = np.diag((1 + 2 * r) * np.ones(nx)) + np.diag(-r * np.ones(nx - 1), 1) + np.diag(-r * np.ones(nx - 1), -1)
A[0, :] = 0
A[0, 0] = 1

# Time-stepping loop
for n in range(1, nt):
    # Explicit scheme
    P_new_explicit = np.zeros(nx)
    for i in range(1, nx - 1):
        P_new_explicit[i] = 2 * P_explicit[i] - P_explicit_prev[i] + r * (P_explicit[i + 1] - 2 * P_explicit[i] + P_explicit[i - 1])
    
    P_explicit_prev = P_explicit.copy()
    P_explicit = P_new_explicit.copy()
    
    # Implicit scheme
    P_implicit_prev = P_implicit.copy()
    P_implicit = np.linalg.solve(A, 2 * P_implicit_prev - P_implicit_prev)

# Plotting results
x = np.linspace(0, L, nx)

plt.figure(figsize=(12, 6))
plt.plot(x, P_explicit, label="Explicit Scheme")
plt.plot(x, P_implicit, label="Implicit Scheme", linestyle='--')
plt.xlabel('Distance (ft)')
plt.ylabel('Pressure (psi)')
plt.title('Pressure Distribution in the Pipe')
plt.legend()
plt.grid()
plt.show()

# Optional: Save the plot or create a video of the pressure distribution over time