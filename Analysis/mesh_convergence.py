# -*- coding: utf-8 -*-
"""
Created on Tue May  6 13:00:17 2025

@author: trott
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt

saveDir = 'C:/Users/trott/Dropbox/Conference Papers/In Progress/Roberts2025_IMECE/Latex/Figures'
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

# Material and geometric properties
E = 210e9        # Young's modulus (Pa)
rho = 7850       # Density (kg/m^3)
width = 0.05     # Width of the beam (m)
thickness = 0.005  # Thickness of the beam (m)
L = 1.0          # Length of the beam (m)
A = width * thickness
I = (width * thickness**3) / 12

# Function to create beam stiffness matrix with full boundary constraints
def create_beam_matrices(n_elements):
    n_nodes = n_elements + 1
    DOF_per_node = 3
    total_DOF = n_nodes * DOF_per_node
    node_positions = np.linspace(0, L, n_nodes)
    
    K = np.zeros((total_DOF, total_DOF))
    midpoint_index = n_nodes // 2
    vertical_index = DOF_per_node * midpoint_index + 1

    for i in range(1, n_elements + 1):
        h = node_positions[i] - node_positions[i - 1]

        # Axial stiffness
        K_axial = E * A / h * np.array([[1, -1], [-1, 1]])

        # Bending stiffness
        K_bend = E * I / h**3 * np.array([
            [12, 6 * h, -12, 6 * h],
            [6 * h, 4 * h**2, -6 * h, 2 * h**2],
            [-12, -6 * h, 12, -6 * h],
            [6 * h, 2 * h**2, -6 * h, 4 * h**2]
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

        # Assemble bending terms
        K[np.ix_([dofs[1], dofs[2], dofs[4], dofs[5]], [dofs[1], dofs[2], dofs[4], dofs[5]])] += K_bend

    # Apply full boundary conditions at both ends (displacement + rotation)
    constrained_dofs = list(range(3)) + list(range(total_DOF - 3, total_DOF))
    for dof in constrained_dofs:
        K[dof, :] = 0
        K[:, dof] = 0
        K[dof, dof] = 1

    return K, vertical_index

# Run convergence study
element_counts = [9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
displacements = []
deltas_theory = (1.0 * L**3) / (192 * E * I)

for n_elem in element_counts:
    try:
        K, mid_idx = create_beam_matrices(n_elem)
        F = np.zeros(K.shape[0])
        F[mid_idx] = 1.0  # Unit vertical force at midpoint
        u = np.linalg.solve(K, F)
        displacements.append(u[mid_idx])
    except np.linalg.LinAlgError:
        displacements.append(np.nan)

# # Display results
import pandas as pd
# # from ace_tools import display_dataframe_to_user

df = pd.DataFrame({
    "Elements": element_counts,
    "Midpoint Displacement (m)": displacements,
    "Theoretical Displacement (m)": [deltas_theory] * len(displacements),
    "Relative Error (%)": [100 * abs((d - deltas_theory) / deltas_theory) if np.isfinite(d) else np.nan for d in displacements]
})

# # display_dataframe_to_user(name="FEM Mesh Convergence Results", dataframe=df)

plt.figure(figsize=(3.6, 2.0), dpi=300)
plt.loglog(element_counts, df["Relative Error (%)"], marker='o', linestyle='None', markersize=4)

plt.xlabel('number of elements', fontname='Times New Roman', fontsize=11)
plt.ylabel('relative error (%)', fontname='Times New Roman', fontsize=11)
plt.xticks(fontname='Times New Roman', fontsize=10)
plt.yticks(fontname='Times New Roman', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.box(True)

plt.tight_layout()
plt.savefig(os.path.join(saveDir, 'fem_convergence_plot.png'), dpi=300)
plt.show()

plt.figure(figsize=(3.6, 2.0), dpi=300)
plt.scatter(element_counts, df["Relative Error (%)"], s=20, color='blue')

plt.xlabel('number of elements', fontname='Times New Roman', fontsize=11)
plt.ylabel('relative error (%)', fontname='Times New Roman', fontsize=11)
plt.xticks(fontname='Times New Roman', fontsize=10)
plt.yticks(fontname='Times New Roman', fontsize=10)
plt.grid(True, linestyle='--', linewidth=0.5)
plt.box(True)

plt.tight_layout()
plt.savefig(os.path.join(saveDir, 'fem_convergence_scatter.png'), dpi=300)
plt.show()