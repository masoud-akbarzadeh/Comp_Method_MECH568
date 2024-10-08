"""
Title: HW1_MECH568:
Author: Masoud Akbarzadeh
Date: 2024-09-06

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
Nx = 10                                        # Number of Spatial points

# Discretization
dx = L / (Nx - 1)

# Parameters
c = 1125    #Speed of the sound for water #Unit: ft/s

# Coefficient
# r = c ** 2 / dx **2 
r = c **2 * dt**2 / dx**2

# Initialize Pressure distribution
y = np.zeros(( Nt + 2, Nx)) # Nt+2 because there is two time boundary at the start
y[0, :] = np.ones(Nx) * 850 # Pressure at time step -1
y[1, :] = np.ones(Nx) * 850 # Pressure at time step 0
for k in range(2, Nt + 2): # 2 instead of 1 and Nt+2 instead of N+1 because there is two time boundary at the start
    y[k, 0] = 0     # Pressure at the start of the pipe
    y[k, -1] = 850  # Pressure at the end of the pipe # in psi


############## Implicit Method - START ################

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

y_implicit = y.copy()
############## Implicit Method - END ################


############## Explicit Method - START ################
 # Time-stepping loop
for k in range(1, Nt+1): #range(1, Nt+1) instead of range(Nt) because there is two time boundary at the start
    for i in range(1, Nx-1):
        y[k + 1, i] = y[k, i-1] * r + y[k, i] * (2-2*r) + y[k, i+1] * r - y[k-1, i]
y_explicit = y.copy()
############## Explicit Method - END ################

# plot the results
X, time = np.meshgrid(np.linspace(0, L, Nx), np.linspace(0, t_max, Nt+2))

# First Plot - Implicit
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(X, time, y_implicit, cmap='viridis')
ax1.set_xlabel('Distance(ft)')
ax1.set_ylabel('Time(s)')
ax1.set_zlabel('Pressure(psi)')
ax1.set_title('Numerical Solution of 1D Wave Equation-implicit')
ax1.view_init(elev=60, azim=145)  # Change the view angle
fig1.savefig('implicit_solution.png')

# Second Plot - Explicit
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(X, time, y_explicit, cmap='viridis')
ax2.set_xlabel('Distance(ft)')
ax2.set_ylabel('Time(s)')
ax2.set_zlabel('Pressure(psi)')
ax2.set_title('Numerical Solution of 1D Wave Equation-explicit')
fig2.savefig('explicit_solution.png')

# Third Plot - Difference
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(X, time, y_explicit - y_implicit, cmap='viridis')
ax3.set_xlabel('Distance(ft)')
ax3.set_ylabel('Time(s)')
ax3.set_zlabel('Pressure(psi)')
ax3.set_title('Difference between Explicit and Implicit Solutions')
fig3.savefig('difference_solution.png')

# Combined Plot
fig_combined = plt.figure()

# First Plot - Implicit
ax_combined1 = fig_combined.add_subplot(131, projection='3d')
ax_combined1.plot_surface(X, time, y_implicit, cmap='viridis')
ax_combined1.set_xlabel('Distance(ft)')
ax_combined1.set_ylabel('Time(s)')
ax_combined1.set_zlabel('Pressure(psi)')
ax_combined1.set_title('Numerical Solution of 1D Wave Equation-implicit')

# Second Plot - Explicit
ax_combined2 = fig_combined.add_subplot(132, projection='3d')
ax_combined2.plot_surface(X, time, y_explicit, cmap='viridis')
ax_combined2.set_xlabel('Distance(ft)')
ax_combined2.set_ylabel('Time(s)')
ax_combined2.set_zlabel('Pressure(psi)')
ax_combined2.set_title('Numerical Solution of 1D Wave Equation-explicit')

# Third Plot - Difference
ax_combined3 = fig_combined.add_subplot(133, projection='3d')
ax_combined3.plot_surface(X, time, y_explicit - y_implicit, cmap='viridis')
ax_combined3.set_xlabel('Distance(ft)')
ax_combined3.set_ylabel('Time(s)')
ax_combined3.set_zlabel('Pressure(psi)')
ax_combined3.set_title('Difference between Explicit and Implicit Solutions')

fig_combined.savefig('combined_solution.png')