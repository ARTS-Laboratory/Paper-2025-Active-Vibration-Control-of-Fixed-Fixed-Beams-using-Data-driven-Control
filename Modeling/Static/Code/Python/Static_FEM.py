# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:13:13 2025

@author: trott
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, solve

# Beam and FEM setup
nElem = 50
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

# Static moment loads
F = np.zeros(nDOF)
moment_nodes = [26, 35]
moment_dofs = [(node - 1) * nDOF_per_node + 2 for node in moment_nodes]
moment_vals = [-800, +800]
for dof, val in zip(moment_dofs, moment_vals):
    F[dof] = val

# Boundary conditions
fixedDOF = [0, 1, 2, nDOF-3, nDOF-2, nDOF-1]
freeDOF = np.setdiff1d(np.arange(nDOF), fixedDOF)

# Reduced system
K_red = K[np.ix_(freeDOF, freeDOF)]
M_red = M[np.ix_(freeDOF, freeDOF)]
F_red = F[freeDOF]

# Static solution
U = np.zeros(nDOF)
U[freeDOF] = solve(K_red, F_red)

# Plot static deflection
x_nodes = np.linspace(0, L, nNode)
v_disp = U[1::3]

plt.figure()
plt.plot(x_nodes, v_disp, 'b-o', linewidth=2, markerfacecolor='r')
plt.xlabel('beam length (m)')
plt.ylabel('vertical deflection (m)')
plt.title('Static Deflection')
plt.grid(True)
plt.show()

# Modal analysis
eigvals, eigvecs = eigh(K_red, M_red)
omega = np.sqrt(eigvals)
freq = omega / (2*np.pi)
print("First 5 Natural Frequencies (Hz):", freq[:5])

# Rayleigh damping
zeta1 = zeta2 = 0.02
w1, w2 = 2*np.pi*freq[0], 2*np.pi*freq[1]
A = np.array([[1/(2*w1), w1/2], [1/(2*w2), w2/2]])
b = np.array([zeta1, zeta2])
alpha, beta = np.linalg.solve(A, b)
C_red = alpha * M_red + beta * K_red

# Newmark-Beta parameters
beta_n = 1/4
gamma = 1/2
dt = 1e-4
Tmax = 0.1
Nt = int(Tmax / dt)
time = np.linspace(0, Tmax, Nt+1)

# Initial conditions
nDOF_red = len(freeDOF)
w = np.zeros((nDOF_red, Nt+1))
v = np.zeros((nDOF_red, Nt+1))
a = np.linalg.solve(M_red, F_red - C_red @ v[:,0] - K_red @ w[:,0])
a = a.reshape(-1, 1)
w[:,0] = w[:,0]
v[:,0] = v[:,0]
a = np.hstack([a] + [np.zeros((nDOF_red, Nt))])

# Newmark constants
a0 = 1 / (beta_n * dt**2)
a1 = gamma / (beta_n * dt)
a2 = 1 / (beta_n * dt)
a3 = 1 / (2*beta_n) - 1
a4 = gamma / beta_n - 1
a5 = dt * (gamma / (2*beta_n) - 1)

Keff = K_red + a0 * M_red + a1 * C_red

# Dynamic loading (impulse)
f_dyn = np.zeros((nDOF_red, Nt+1))
impulse_node = (nElem // 2) + 18
impulse_dof_global = (impulse_node - 1) * nDOF_per_node + 1
impulse_dof_local = np.where(freeDOF == impulse_dof_global)[0][0]
impulse_val = 1000
impulse_steps = int(0.001 / dt)
f_dyn[impulse_dof_local, :impulse_steps] = impulse_val

# Time integration loop
for n in range(Nt):
    Feff = f_dyn[:,n+1] + \
           M_red @ (a0*w[:,n] + a2*v[:,n] + a3*a[:,n]) + \
           C_red @ (a1*w[:,n] + a4*v[:,n] + a5*a[:,n])
    w[:,n+1] = solve(Keff, Feff)
    a[:,n+1] = a0*(w[:,n+1] - w[:,n]) - a2*v[:,n] - a3*a[:,n]
    v[:,n+1] = v[:,n] + dt*((1-gamma)*a[:,n] + gamma*a[:,n+1])

# Plot dynamic displacement at impulse DOF
plt.figure()
plt.plot(time, w[impulse_dof_local,:], linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Vertical displacement at midpoint (m)')
plt.title('Dynamic response to impulse force (Newmark-Beta)')
plt.grid(True)
plt.show()
