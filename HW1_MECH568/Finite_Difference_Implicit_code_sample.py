import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1e-3        # Thickness of the artery in m
t_max = 3.0     # Total simulation time
Nx = 10         # Number of spatial points
Nt = 3          # Number of time steps
K = 0.585       # Thermal conductivity of artery
rho = 1000      # Artery Density
c_p = 1200      # Specific Heat of the Artery

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
    T[k, -1] = 25 + (170 - 25) * k / (Nt + 1)

# Generate coefficient Matrix
KK = np.zeros((Nx - 2, Nx - 2))
for i in range(Nx - 2):
    KK[i, i] = 2 * r + 1
    if i != Nx - 3:
        KK[i, i + 1] = -r
    if i != 0:
        KK[i, i - 1] = -r

# Solve
for k in range(Nt):
    # Generate External Boundary condition Vector
    T_ext = np.zeros(Nx - 2)
    T_ext[0] = r * T[k + 1, 0]
    T_ext[-1] = r * T[k + 1, -1]

    # Solve via Direct Method (NumPy's linalg.solve)
    T[k + 1, 1:Nx - 1] = np.linalg.solve(KK, T[k, 1:Nx - 1] + T_ext)

# Plot the results
X, t_max = np.meshgrid(np.linspace(0, L, Nx), np.linspace(0, t_max, Nt + 1))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, t_max, T.T)
ax.set_xlabel('Distance')
ax.set_ylabel('Time')
ax.set_zlabel('Temperature')
ax.set_title('Numerical Solution of 1D Heat Equation')
plt.show()
