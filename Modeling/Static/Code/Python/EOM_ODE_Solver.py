# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:17:17 2025

@author: trott

Beam Vibration Simulation using FEM and Impulse Response (Python)

Description:
------------
This script simulates the dynamic response of a fixed-fixed Euler-Bernoulli beam 
subjected to an impulsive transverse force using the Finite Element Method (FEM). 
The beam is discretized into multiple elements with 3 degrees of freedom per node 
(axial displacement, vertical displacement, and rotation). 

Key Features:
-------------
- Consistent mass and stiffness matrix assembly for axial and bending behavior.
- Fixed boundary conditions applied to both beam ends.
- Modal analysis performed to extract natural frequencies.
- Rayleigh damping is constructed to match a 2% damping ratio on the first two modes.
- An impulsive vertical force is applied at an internal node.
- Equation of motion is solved using the `Radau` stiff ODE solver.
- LU factorization is used to efficiently solve the mass matrix inverse repeatedly.
- Final output shows vertical displacement at the midpoint node over time.

Usage Notes:
------------
Due to the large system size and stiffness of the beam, this simulation may take
significant time to run for high element counts (e.g., nElem = 50). Use a smaller
value for interactive testing (e.g., nElem = 10) or run the full-resolution simulation
on a machine with more computational resources.

Dependencies:
-------------
- numpy
- matplotlib
- scipy.linalg
- scipy.integrate

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
from scipy.linalg import lu_factor, lu_solve

# FEM parameters
nElem = 50  # Full size
E = 210e9
A = 0.01
I = 8.333e-6
rho = 7850
L = 2.0
Le = L / nElem
nNode = nElem + 1
nDOF_per_node = 3
nDOF = nNode * nDOF_per_node

# Element matrices
ke_axial = (E * A / Le) * np.array([[1, -1], [-1, 1]])
ke_bending = (E * I / Le**3) * np.array([
    [12, 6*Le, -12, 6*Le],
    [6*Le, 4*Le**2, -6*Le, 2*Le**2],
    [-12, -6*Le, 12, -6*Le],
    [6*Le, 2*Le**2, -6*Le, 4*Le**2]
])
me_axial = (rho * A * Le / 6) * np.array([[2, 1], [1, 2]])
me_bending = (rho * A * Le / 420) * np.array([
    [156, 22*Le, 54, -13*Le],
    [22*Le, 4*Le**2, 13*Le, -3*Le**2],
    [54, 13*Le, 156, -22*Le],
    [-13*Le, -3*Le**2, -22*Le, 4*Le**2]
])

# Global assembly
K = np.zeros((nDOF, nDOF))
M = np.zeros((nDOF, nDOF))
for e in range(nElem):
    idx = np.arange(e * nDOF_per_node, e * nDOF_per_node + 6)
    ke = np.zeros((6,6))
    me = np.zeros((6,6))
    axial_dofs = [0, 3]
    bending_dofs = [1, 2, 4, 5]
    ke[np.ix_(axial_dofs, axial_dofs)] = ke_axial
    ke[np.ix_(bending_dofs, bending_dofs)] = ke_bending
    me[np.ix_(axial_dofs, axial_dofs)] = me_axial
    me[np.ix_(bending_dofs, bending_dofs)] = me_bending
    K[np.ix_(idx, idx)] += ke
    M[np.ix_(idx, idx)] += me

# Boundary conditions
fixedDOF = [0, 1, 2, nDOF-3, nDOF-2, nDOF-1]
freeDOF = np.setdiff1d(np.arange(nDOF), fixedDOF)
K_red = K[np.ix_(freeDOF, freeDOF)]
M_red = M[np.ix_(freeDOF, freeDOF)]

# Rayleigh damping
eigvals, _ = eigh(K_red, M_red)
omega = np.sqrt(eigvals)
freq = omega / (2*np.pi)
zeta1 = zeta2 = 0.02
w1, w2 = 2*np.pi*freq[0], 2*np.pi*freq[1]
A_damp = np.array([[1/(2*w1), w1/2], [1/(2*w2), w2/2]])
b_damp = np.array([zeta1, zeta2])
alpha, beta = np.linalg.solve(A_damp, b_damp)
C_red = alpha * M_red + beta * K_red

# Impulse setup
impulse_node = (nElem // 2) + 1
impulse_dof_global = (impulse_node - 1) * nDOF_per_node + 1
impulse_dof_local = np.where(freeDOF == impulse_dof_global)[0][0]
impulse_val = 1000
impulse_time = 0.001

# Pre-factorize mass matrix
lu_M, piv = lu_factor(M_red)

# Define ODE function using LU solve
def eom_lu_full(t, y):
    n = len(M_red)
    u = y[:n]
    v = y[n:]
    f = np.zeros(n)
    if t < impulse_time:
        f[impulse_dof_local] = impulse_val
    a = lu_solve((lu_M, piv), f - C_red @ v - K_red @ u)
    return np.concatenate([v, a])

# Initial state and solver setup
y0 = np.zeros(2 * len(freeDOF))
t_span = (0, 0.1)
t_eval = np.linspace(*t_span, 1001)

# Solve using Radau
sol_full = solve_ivp(eom_lu_full, t_span, y0, t_eval=t_eval, method='Radau')

# Plot the midpoint vertical displacement
plt.figure()
plt.plot(sol_full.t, sol_full.y[impulse_dof_local,:], linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Vertical displacement at midpoint (m)')
plt.title('Dynamic response (nElem=50, Radau + LU)')
plt.grid(True)
plt.show()