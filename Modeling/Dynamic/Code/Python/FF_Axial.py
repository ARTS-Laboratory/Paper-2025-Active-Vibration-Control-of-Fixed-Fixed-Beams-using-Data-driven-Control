# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 11:25:12 2025

@author: trott
"""

# %% Imports
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import matplotlib.pyplot as plt

# %% Beam properties
E = 210e9  # Young's modulus (Pa)
rho = 7850  # Density (kg/m^3)
width = 0.05  # Beam width (m)
thickness = 0.005  # Beam thickness (m)
L = 1.0  # Beam length (m)
A = width * thickness  # Cross-sectional area
I = (width * thickness**3) / 12  # Moment of inertia

# %% Simulation parameters
n_nodes = 50  # Number of nodes (discretization)
n_elements = n_nodes - 1
node_positions = np.linspace(0, L, n_nodes)
dt = 0.001  # Time step (s)
t_total = 3  # Total simulation time (s)
n_steps = int(t_total / dt)
impact_time = 0.2  # Impact start time (s)
impact_duration = 0.001  # Impact force duration (s)
impact_magnitude = 1000  # Impact force (N)
impact_node = n_nodes // 2  # Midpoint node

# %% DOFs and indexing
DOF_per_node = 3  # axial (x), vertical (y), rotation
total_DOF = DOF_per_node * n_nodes
midpoint_node = n_nodes // 2
midpoint_y_index = DOF_per_node * impact_node + 1  # vertical DOF at midpoint

# %% Initialize matrices
K = np.zeros((total_DOF, total_DOF))
M = np.zeros((total_DOF, total_DOF))

for i in range(1, n_elements + 1):
    h = node_positions[i] - node_positions[i - 1]

    # Axial stiffness and mass
    K_axial = E * A / h * np.array([[1, -1], [-1, 1]])
    M_axial = rho * A * h / 6 * np.array([[2, 1], [1, 2]])

    # Bending stiffness and mass
    K_bend = E * I / h**3 * np.array([
        [12, 6 * h, -12, 6 * h],
        [6 * h, 4 * h**2, -6 * h, 2 * h**2],
        [-12, -6 * h, 12, -6 * h],
        [6 * h, 2 * h**2, -6 * h, 4 * h**2]
    ])
    M_bend = rho * A * h / 420 * np.array([
        [156, 22 * h, 54, -13 * h],
        [22 * h, 4 * h**2, 13 * h, -3 * h**2],
        [54, 13 * h, 156, -22 * h],
        [-13 * h, -3 * h**2, -22 * h, 4 * h**2]
    ])

    # Global DOF indices
    n1 = i - 1
    n2 = i
    dofs = [
        3 * n1,     # axial n1
        3 * n1 + 1, # vertical n1
        3 * n1 + 2, # rotation n1
        3 * n2,     # axial n2
        3 * n2 + 1, # vertical n2
        3 * n2 + 2  # rotation n2
    ]

    # Assemble axial terms
    K[np.ix_([dofs[0], dofs[3]], [dofs[0], dofs[3]])] += K_axial
    M[np.ix_([dofs[0], dofs[3]], [dofs[0], dofs[3]])] += M_axial

    # Assemble bending terms
    K[np.ix_([dofs[1], dofs[2], dofs[4], dofs[5]], [dofs[1], dofs[2], dofs[4], dofs[5]])] += K_bend
    M[np.ix_([dofs[1], dofs[2], dofs[4], dofs[5]], [dofs[1], dofs[2], dofs[4], dofs[5]])] += M_bend

# %% Boundary conditions (Fixed-Fixed)
# Left end
K[:3, :] = 0
K[:, :3] = 0
K[0, 0] = K[1, 1] = K[2, 2] = 1

M[:3, :] = 0
M[:, :3] = 0
M[0, 0] = M[1, 1] = M[2, 2] = 1

# Right end
K[-3:, :] = 0
K[:, -3:] = 0
K[-3, -3] = K[-2, -2] = K[-1, -1] = 1

M[-3:, :] = 0
M[:, -3:] = 0
M[-3, -3] = M[-2, -2] = M[-1, -1] = 1

# %% Static test
# Apply vertical unit force at an unconstrained DOF (midpoint node is safe)
midpoint_node = n_nodes // 2
midpoint_vertical_index = DOF_per_node * midpoint_node + 1

F_static = np.zeros(total_DOF)
F_static[midpoint_vertical_index] = 1.0  # Unit load at midpoint

# Solve Ku = F for static displacement
u_static = np.linalg.solve(K, F_static)

# Get vertical displacement at midpoint
midpoint_disp = u_static[midpoint_vertical_index]
print(f"Static midpoint displacement (numerical): {midpoint_disp:.6e} m")

# Compare with Euler-Bernoulli theory (fixed-fixed, load at center):
# δ = (F * L^3) / (192 * E * I)
delta_theory = (1.0 * L**3) / (192 * E * I)
print(f"Static midpoint displacement (theoretical): {delta_theory:.6e} m")

# %% Diagnostics
cond_K = np.linalg.cond(K)
print(f"Condition number of stiffness matrix K: {cond_K:.2e}")

# %% Damping and Effective Matrix
alpha = 1e-5
beta = 1e-6
C = alpha * M + beta * K

gamma = 0.5
beta_n = 0.25
K_eff = K + gamma / (beta_n * dt) * C + 1 / (beta_n * dt**2) * M

try:
    K_inv = np.linalg.inv(K_eff)
    print("K_eff successfully inverted.")
except np.linalg.LinAlgError:
    print("Error: K_eff is singular and cannot be inverted.")

# %% Initialize states
displacements = np.zeros((n_steps, total_DOF))
velocities = np.zeros((n_steps, total_DOF))

if np.linalg.cond(M) > 1e7:
    M /= np.max(np.abs(M))

# %% Simulation loop
for step in range(1, n_steps):
    t = step * dt
    F = np.zeros(total_DOF)

    # Apply impact force vertically at midpoint node
    if impact_time <= t < impact_time + impact_duration:
        F[midpoint_y_index] = impact_magnitude
        print(f"Impact applied at midpoint (node {impact_node}) with magnitude {impact_magnitude} at step {step}")

    # Compute effective force
    F_eff = (F +
             M @ ((1 / (beta_n * dt**2)) * displacements[step - 1] +
                  (1 / (beta_n * dt)) * velocities[step - 1] +
                  ((0.5 / beta_n) - 1) * np.zeros(total_DOF)) +
             C @ ((gamma / (beta_n * dt)) * displacements[step - 1] +
                  (gamma / beta_n - 1) * velocities[step - 1] +
                  dt * ((gamma / (2 * beta_n)) - 1) * np.zeros(total_DOF)))

    # Solve for new displacement and velocity
    displacements_new = K_inv @ F_eff
    velocities_new = (displacements_new - displacements[step - 1]) / dt

    # Store new state
    displacements[step, :] = displacements_new
    velocities[step, :] = velocities_new

    # print(f"Step {step}: Midpoint Vertical Displacement = {displacements_new[midpoint_y_index]:.6e}")


# %% Plot colors
cmap = plt.get_cmap("tab10")

# Define colors
uncontcolor = cmap(7)  # Gray
impactcolor = cmap(0)  # Blue

# %% Midpoint Displacement Plot
plt.rc('font', family='Times New Roman', size=10)
plt.figure(figsize=(6.5, 3))
time_array = np.linspace(0, t_total, n_steps)
midpoint_displacement = displacements[:, midpoint_y_index]



plt.plot(time_array, midpoint_displacement, label="midpoint displacement", color=impactcolor, linewidth=1.0)


plt.xlabel("time (s)")
plt.ylabel("displacement (m)")
plt.grid()
plt.legend(facecolor="white", edgecolor="lightgray", framealpha=1, frameon=True)

# # Save the plot
# save_dir = r"C:\Users\trott\OneDrive\Documents\University of South Carolina\Research\SPIE 2025 SSNDE\Prev_sim"
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)  # Create the directory if it doesn't exist
# save_path = os.path.join(save_dir, "midpoint_displacement_plot.png")
# plt.savefig(save_path, dpi=300, bbox_inches='tight')

# plt.show()


# Print mass matrix properties
print("Max mass matrix value:", np.max(M))
print("Min mass matrix value:", np.min(M))
print("Condition number of M:", np.linalg.cond(M))
