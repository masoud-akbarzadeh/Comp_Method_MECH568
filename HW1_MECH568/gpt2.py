import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
c = 1125  # Speed of sound in water (ft/s)
L = 100   # Length of the pipe (ft)
dx = 10    # Spatial step (ft)
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

# Arrays to store the previous time step and full pressure field
P_explicit_prev = P_explicit.copy()
P_implicit_prev = P_implicit.copy()

# Store results over time for plotting
P_explicit_time = np.zeros((nt, nx))
P_implicit_time = np.zeros((nt, nx))

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
    P_explicit_time[n, :] = P_explicit
    
    # Implicit scheme
    P_implicit_prev = P_implicit.copy()
    P_implicit = np.linalg.solve(A, 2 * P_implicit_prev - P_implicit_prev)
    P_implicit_time[n, :] = P_implicit

# Creating meshgrid for plotting
X, time = np.meshgrid(np.linspace(0, L, nx), np.linspace(0, T, nt))

# Plotting results for Explicit Scheme
fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X, time, P_explicit_time, cmap='viridis')
ax.set_xlabel('Distance (ft)')
ax.set_ylabel('Time (s)')
ax.set_zlabel('Pressure (psi)')
ax.set_title('Explicit Scheme: Pressure Distribution')

# Plotting results for Implicit Scheme
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, time, P_implicit_time, cmap='viridis')
ax2.set_xlabel('Distance (ft)')
ax2.set_ylabel('Time (s)')
ax2.set_zlabel('Pressure (psi)')
ax2.set_title('Implicit Scheme: Pressure Distribution')

plt.show()
