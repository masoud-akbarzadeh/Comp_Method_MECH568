"""
Title: HW1_MECH568:
Author: Masoud Akbarzadeh
Date: 2024-08-00
Description: [Brief description of what the script does]

Usage: 
- 
- 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Variables
# y is the is P or pressure in psi
# x is the distance from the start of the pipe in ft
# t is time in sec


# y[2, :] Pressure at time step 1 # the first unknown Pressure
# y[1, :] Pressure at time step 0
# y[0, :] Pressure at time step -1  

t_min = 0    # time at the end               # in sec 
t_max = 0.05     # time at the start               # in sec
dt = 0.001  # delta_t or time increaments       # in sec
Nt = int((t_max - t_min) / dt) # Number of time steps

x_min = 0                                       # in ft
x_max = 100     # the length at the end         # in ft
L = x_max - x_min
Nx = 100                                        # Number of Spatial points

# Discretization
dx = L / (Nx - 1)

# Parameters
c = 1125    #Speed of the sound for water #Unit: ft/s

# Coefficient
r = c ** 2 / dx **2 

print(f"{Nt} + {type(Nt)}")
print(f"{Nx} + {type(Nx)}")
# Initialize Pressure distribution
y = np.zeros(( Nt + 2, Nx)) # Nt+2 because there is two time boundary at the start
y[0, :] = np.ones(Nx) * 850 # Pressure at time step -1
y[1, :] = np.ones(Nx) * 850 # Pressure at time step 0
for k in range(2, Nt + 2): # 2 instead of 1 and Nt+2 instead of N+1 because there is two time boundary at the start
    y[k, 0] = 0     # Pressure at the start of the pipe
    y[k, -1] = 850  # Pressure at the end of the pipe # in psi


############## Implicit Method ################

# Generate coefficient Matrix
KK = np.zeros((Nx - 2, Nx - 2))
for i in range(Nx - 2):
    KK[i, i] = r + 1/2
    if i != Nx - 3:
        KK[i, i + 1] = -r /2
    if i != 0:
        KK[i, i - 1] = -r /2


# Solve
for k in range(1, Nt +1):
# for k in range(Nt + 1): # +1 because there is two time boundary at the start
    y_ext = np.zeros(Nx - 2) #np.ones(Nx - 2) * - (1/2) * y[k - 1,]
    y_ext[0] = r / 2 * y[k + 1, 0] 
    y_ext[-1] = r / 2 * y[k + 1, -1]

    y[k + 1, 1:Nx -1] = np.linalg.solve(KK, y[k, 1:Nx - 1] -(1/2)*y[k-1, 1:Nx - 1] + y_ext )

y_implicit = y
############## Implicit Method ################



############## Explicit Method ################


# plot the results
X, time = np.meshgrid(np.linspace(0, L, Nx), np.linspace(0, t_max, Nt+2))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, time, y) # don't know why there should be no .T after y ???
ax.set_xlabel('Distance')
ax.set_ylabel('Time')
ax.set_zlabel('Pressure')
ax.set_title('Numerical Solution of 1D Wave Equation')
print(f"KK:{KK}")
print(f"P:{y}")
plt.show()
