import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1e-3            # Thickness of the artery in m
t_max = 3.0         # Total simulation time
Nx = 10             # Number of spatial points
Nt = 300            # Number of time steps
K = 0.585           # Thermal conductivity of artery
rho = 1000          # Artery Density
c_p = 1200          # Specific Heat of the Artery

# Discretization
dx = L / (Nx - 1)
dt = t_max / Nt

# Coefficient
r = K / (rho * c_p) * dt / dx**2

# Initialize temperature distribution
T = np.zeros((Nt + 1, Nx))
T[0, :] = np.ones(Nx) * 25
for k in range(1, Nt + 1):
    T[k, 0] = 25 + (170 - 25) * k / (Nt + 1)
    T[k, -1] = 25 + (170 - 25) * k / Nt

# Time-stepping loop
for k in range(Nt):
    for i in range(1, Nx - 1):
        T[k + 1, i] = T[k, i] + r * (T[k, i - 1] - 2 * T[k, i] + T[k, i + 1])

# Plot the results
X, t_max = np.meshgrid(np.arange(t_max + dt, step=dt), np.arange(L + dx, step=dx))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X[:,0:300], t_max[:,0:300], T[1:, :].T, cmap='viridis')
ax.set_xlabel('Time')
ax.set_ylabel('Distance')
ax.set_zlabel('Temperature')
ax.set_title('Numerical Solution of 1D Heat Equation')
plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
