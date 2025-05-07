# -*- coding: utf-8 -*-
"""
Created on Tue May  6 10:40:46 2025

@author: trott
"""

# %% Imports
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

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
dt = 0.0005  # Time step (s)
t_total = 3  # Total simulation time (s)
n_steps = int(t_total / dt)
impact_time = 0.1  # Impact start time (s)
impact_duration = 0.001  # Impact force duration (s)
impact_magnitude = 1000  # Impact force (N)
impact_node = n_nodes // 2  # Midpoint node
moment_scaling_factor = 1

# %% DOFs and indexing
DOF_per_node = 3  # axial (x), vertical (y), rotation
total_DOF = DOF_per_node * n_nodes
midpoint_y_index = DOF_per_node * impact_node + 1  # vertical DOF at midpoint
midpoint_node = n_nodes // 2
midpoint_axial_index = 3 * midpoint_node
midpoint_vertical_index = DOF_per_node * midpoint_node + 1
control_node = n_nodes // 4
node_1 = control_node         # left of actuator
node_2 = control_node + 1     # right of actuator
control_axial_index = 3 * control_node

axial_index_1 = 3 * node_1
axial_index_2 = 3 * node_2
vertical_index_1 = 3 * node_1 + 1
vertical_index_2 = 3 * node_2 + 1
control_vertical_index = DOF_per_node * control_node + 1
rotation_index_1 = DOF_per_node * node_1 + 2  # rotation DOF at node_1
rotation_index_2 = DOF_per_node * node_2 + 2  # rotation DOF at node_2

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

# %% LQG Setup
A_aug = np.zeros((2 * total_DOF, 2 * total_DOF))
A_aug[:total_DOF, total_DOF:] = np.eye(total_DOF)
A_aug[total_DOF:, :total_DOF] = -np.linalg.inv(M) @ K
A_aug[total_DOF:, total_DOF:] = -np.linalg.inv(M) @ C

B_aug = np.zeros((2 * total_DOF, 1))
B_aug[total_DOF + vertical_index_1] = -moment_scaling_factor * thickness / 2
B_aug[total_DOF + vertical_index_2] = moment_scaling_factor * thickness / 2

# Q = np.eye(2 * total_DOF) * 1e-1
# R = np.array([[0.01]])
Q = np.zeros((2 * total_DOF, 2 * total_DOF))
# Moderate penalty across all DOFs
Q = np.eye(2 * total_DOF) * 1e0  # light regularization
Q[midpoint_vertical_index, midpoint_vertical_index] = 1e7  # vertical displacement
Q[total_DOF + midpoint_vertical_index, total_DOF + midpoint_vertical_index] = 1e5 # vertical velocity

# Raise control effort penalty to reduce jitter
R = np.array([[0.4]])

P = solve_continuous_are(A_aug, B_aug, Q, R)
K_lqr = np.linalg.inv(R) @ B_aug.T @ P

# Observation matrix: only midpoint vertical acceleration
C_accel = np.zeros((1, 2 * total_DOF))
C_accel = np.zeros((3, 2 * total_DOF))
C_accel[0, midpoint_vertical_index] = 1.0
C_accel[1, total_DOF + midpoint_vertical_index] = 1.0
C_accel[2, total_DOF + vertical_index_1] = 1.0

# Estimator noise covariances
W = np.eye(2 * total_DOF) * 1e-6  # Process noise
V = np.eye(3) * 1e-2  # Measurement noise

# Kalman Gain (observer)
P_kf = solve_continuous_are(A_aug.T, C_accel.T, W, V)
L = P_kf @ C_accel.T @ np.linalg.inv(V)

# Initialize state estimate
x_hat = np.zeros((2 * total_DOF, 1))

control_force_lqg = 0.0

# %% Sim Variables
# Initialize displacement and velocity arrays
displacements_uncontrolled = np.zeros((n_steps, total_DOF))
velocities_uncontrolled = np.zeros((n_steps, total_DOF))
displacements_pd = np.zeros((n_steps, total_DOF))
velocities_pd = np.zeros((n_steps, total_DOF))
displacements_mlp = np.zeros((n_steps, total_DOF))
velocities_mlp = np.zeros((n_steps, total_DOF))
accelerations_uncontrolled = np.zeros((n_steps, total_DOF))
accelerations_pd = np.zeros((n_steps, total_DOF))
accelerations_mlp = np.zeros((n_steps, total_DOF))
displacements_lqg = np.zeros((n_steps, total_DOF))
velocities_lqg = np.zeros((n_steps, total_DOF))
accelerations_lqg = np.zeros((n_steps, total_DOF))


# Scale M to prevent numerical instability if necessary
if np.linalg.cond(M) > 1e7:
    M /= np.max(np.abs(M))
    
# Simulation variables
state_variables = ['displacements', 'velocities', 'accelerations']
displacement_history_uncontrolled = np.zeros((n_steps, total_DOF))
displacement_history_pd = np.zeros((n_steps, total_DOF))
displacement_history_mlp = np.zeros((n_steps, total_DOF))
displacement_history_lqg = np.zeros((n_steps, total_DOF))
# Initialize states for all controllers
def initialize_states():
    return {var: np.zeros(total_DOF) for var in state_variables}

states = {
    'uncontrolled': initialize_states(),
    'pd': initialize_states(),
    'mlp': initialize_states()
}

# %% PD Controller Setup
kp = 50  # Proportional gain
kd = 0.001  # Derivative gain

MAX_FORCE = 5000
# Store training data from PD controller
lqg_training_data = []

pd_forces = np.zeros(n_steps)

# %% Simulation Loop (Uncontrolled and PD)
for step in range(1, n_steps):
    t = step * dt
    F = np.zeros(total_DOF)

    # Apply impact force over the specified duration
    if impact_time <= t < impact_time + impact_duration:
        F[midpoint_vertical_index] = impact_magnitude

    # Clone the base force vector for each controller to isolate their actions
    F_uncontrolled_base = np.copy(F)
    F_pd_base = np.copy(F)
    # F_mlp_base = np.copy(F)

    # ------------------- Uncontrolled Response -------------------
    F_uncontrolled = F_uncontrolled_base + M @ ((1 / (beta_n * dt**2)) * displacements_uncontrolled[step - 1] +
                              (1 / (beta_n * dt)) * velocities_uncontrolled[step - 1] +
                              ((0.5 / beta_n) - 1) * np.zeros(total_DOF)) + \
                      C @ ((gamma / (beta_n * dt)) * displacements_uncontrolled[step - 1] +
                          (gamma / beta_n - 1) * velocities_uncontrolled[step - 1] +
                          dt * ((gamma / (2 * beta_n)) - 1) * np.zeros(total_DOF))
    
    
    # Solve for uncontrolled displacements and velocities
    displacements_uncontrolled_new = K_inv @ F_uncontrolled
    velocities_uncontrolled_new = (displacements_uncontrolled_new - displacements_uncontrolled[step - 1]) / dt

    # Store uncontrolled response in the dedicated array
    accelerations_uncontrolled[step, :] = np.linalg.solve(
        M, F - C @ velocities_uncontrolled_new - K @ displacements_uncontrolled_new)
    displacements_uncontrolled[step, :] = displacements_uncontrolled_new
    velocities_uncontrolled[step, :] = velocities_uncontrolled_new
    displacement_history_uncontrolled[step, :] = displacements_uncontrolled_new
    
    
    # ---------------------- PD Control ----------------------
    # F_pd = M @ ((1 / (beta_n * dt**2)) * displacements_pd[step - 1] +
    #                       (1 / (beta_n * dt)) * velocities_pd[step - 1] +
    #                       ((0.5 / beta_n) - 1) * np.zeros(total_DOF)) + \
    #                   C @ ((gamma / (beta_n * dt)) * displacements_pd[step - 1] +
    #                       (gamma / beta_n - 1) * velocities_pd[step - 1] +
    #                       dt * ((gamma / (2 * beta_n)) - 1) * np.zeros(total_DOF))
                     

    # # Solve for uncontrolled displacements and velocities
    # displacements_pd_new = K_inv @ F_pd
    # velocities_pd_new = (displacements_pd_new - displacements_pd[step - 1]) / dt
    # accelerations_pd[step - 1, :] = np.linalg.solve(
    #     M, F_pd - C @ velocities_pd[step - 1] - K @ displacements_pd[step - 1])
    
    # displacements_pd[step, :] = displacements_pd_new
    # velocities_pd[step, :] = velocities_pd_new

    # PD control at control node based on axial DOF acceleration
    if step == 1:
        prev_derivative = 0.0  # Initialize once at the start
    
    # Predict system response without control for current timestep
    F_temp = np.zeros(total_DOF)
    F_temp[midpoint_vertical_index] = impact_magnitude if impact_time <= t < impact_time + impact_duration else 0.0
    
    # Estimate acceleration at midpoint vertical DOF
    acc_est = np.linalg.solve(M, F_temp - C @ velocities_pd[step - 1] - K @ displacements_pd[step - 1])
    acc_curr = acc_est[midpoint_vertical_index]
    acc_prev = accelerations_pd[step - 2, midpoint_vertical_index] if step > 2 else 0.0
    
    # PD error on vertical acceleration, driving axial actuator
    error = -acc_curr
    derivative_raw = -(acc_curr - acc_prev) / dt
    
    # Optional smoothing filter (low-pass) for derivative
    alpha = 0.1  # Tuning parameter (0 = sharp, 1 = smooth)
    derivative = alpha * derivative_raw + (1 - alpha) * prev_derivative
    prev_derivative = derivative  # Store for next iteration
    
    # if step % 100 == 0:
    #     print(f"Step {step}: acc = {acc_curr:.4e}, acc_prev = {acc_prev:.4e}, deriv = {derivative:.4e}")
    
    # Compute PD control force
    control_force_pd = kp * error + kd * derivative
    
    # Clamp control force
    if not np.isfinite(control_force_pd):
        print(f"[PD] NaN or Inf detected at step {step}, skipping...")
        control_force_pd = 0.0
    
    control_force_pd = np.clip(control_force_pd, -MAX_FORCE, MAX_FORCE)
    
    pd_forces[step] = control_force_pd
    
    # if step % 100 == 0:
    #     print(f"PD Force = {control_force_pd:.4f}")
    
    # # Log for debugging
    # if step % 500 == 0:
    #     print(f"Step {step}: error = {error:.4e}, derivative = {derivative:.4e}, PD force = {control_force_pd:.4e}")
    
    # Piezoelectric control forces at adjacent nodes
    # F_pd_base[axial_index_1] += control_force_pd   # +N
    # F_pd_base[axial_index_2] -= control_force_pd   # -N
    
    # Bending moment via vertical force couple (±N * h/2)
    control_moment_pd = moment_scaling_factor * control_force_pd * (thickness / 2)
    F_pd_base[rotation_index_1] += control_moment_pd
    F_pd_base[rotation_index_2] -= control_moment_pd  
    
    # Final effective force
    F_pd_eff = F_pd_base + M @ ((1 / (beta_n * dt**2)) * displacements_pd[step - 1] +
                                (1 / (beta_n * dt)) * velocities_pd[step - 1] +
                                ((0.5 / beta_n) - 1) * np.zeros(total_DOF)) + \
               C @ ((gamma / (beta_n * dt)) * displacements_pd[step - 1] +
                    (gamma / beta_n - 1) * velocities_pd[step - 1] +
                    dt * ((gamma / (2 * beta_n)) - 1) * np.zeros(total_DOF))
    
    # New state
    displacements_pd_new = K_inv @ F_pd_eff
    velocities_pd_new = (displacements_pd_new - displacements_pd[step - 1]) / dt
    accelerations_pd[step, :] = np.linalg.solve(
        M, F_pd_eff - C @ velocities_pd_new - K @ displacements_pd_new)
    
    displacements_pd[step, :] = displacements_pd_new
    velocities_pd[step, :] = velocities_pd_new
    displacement_history_pd[step, :] = displacements_pd_new
    
    # if step % 500 == 0:
    #     print(f"Step {step}: PD control force = {control_force_pd:.2f}, Moment = {control_moment_pd:.2f}")
        
    # if step % 500 == 0:
    #     print(f"Step {step}: error = {error:.4e}, derivative = {derivative:.4e}, PD force = {control_force_pd:.4e}")

# %% Simulation Loop (LQG)
# Initialize storage for LQG forces outside the loop
lqg_forces = np.zeros(n_steps)

for step in range(1, n_steps):
    t = step * dt
    F_lqg_base = np.zeros(total_DOF)

    # ----- Measurement from system -----
    y_measured = accelerations_lqg[step - 1, midpoint_vertical_index]  # measurement from previous step
    y_hat = C_accel @ x_hat
    if not np.isfinite(y_hat).all():
        print(f"[LQG] Warning: Invalid y_hat at step {step}")
        y_hat = np.array([[0.0]])

    # ----- Kalman Correction -----
    L_eff = L * np.exp(-t / 1)
    x_hat += dt * (L_eff @ (y_measured - y_hat))
    x_hat = np.tanh(x_hat / 500.0) * 500.0

    # ----- LQR Control -----
    u_lqg = -K_lqr @ x_hat
    control_force_lqg = u_lqg.item()
    if not np.isfinite(control_force_lqg):
        print(f"[LQG] Warning: NaN/Inf control at step {step}, zeroing it")
        control_force_lqg = 0.0
    control_force_lqg = np.clip(control_force_lqg, -MAX_FORCE, MAX_FORCE)

    # Piezoelectric control forces at adjacent nodes
    # F_lqg_base[axial_index_1] += control_force_lqg   # +N
    # F_lqg_base[axial_index_2] -= control_force_lqg   # -N
    
    control_moment_lqg = moment_scaling_factor * control_force_lqg * (thickness / 2)
    F_lqg_base[rotation_index_1] += control_moment_lqg
    F_lqg_base[rotation_index_2] -= control_moment_lqg  
    

    # Apply impact force if within window
    if impact_time <= t < impact_time + impact_duration:
        F_lqg_base[midpoint_vertical_index] += impact_magnitude

    # Compute effective force for Newmark-beta
    F_lqg_eff = F_lqg_base + M @ ((1 / (beta_n * dt**2)) * displacements_lqg[step - 1] +
                                  (1 / (beta_n * dt)) * velocities_lqg[step - 1]) + \
                C @ ((gamma / (beta_n * dt)) * displacements_lqg[step - 1] +
                     (gamma / beta_n - 1) * velocities_lqg[step - 1])

    # Solve dynamics
    displacements_lqg_new = K_inv @ F_lqg_eff
    velocities_lqg_new = (displacements_lqg_new - displacements_lqg[step - 1]) / dt
    accelerations_lqg[step, :] = np.linalg.solve(
        M, F_lqg_eff - C @ velocities_lqg_new - K @ displacements_lqg_new)

    # Save to state history
    displacements_lqg[step, :] = displacements_lqg_new
    velocities_lqg[step, :] = velocities_lqg_new
    displacement_history_lqg[step, :] = displacements_lqg_new
    lqg_forces[step] = control_force_lqg

    # Save training data
    lqg_training_data.append((
        displacements_lqg[step, control_vertical_index],
        velocities_lqg[step, control_vertical_index],
        accelerations_lqg[step, control_vertical_index],
        control_force_lqg))

    # # Debug output
    # if step % 1000 == 0:
    #     print(f"[LQG] Step {step}, ||x̂|| = {np.linalg.norm(x_hat):.2e}, "
    #           f"Force = {control_force_lqg:.2f}, y_meas = {y_measured:.2e}, "
    #           f"Midpoint disp = {displacements_lqg[step, midpoint_vertical_index]:.3e}")
    #     disp_mean = np.mean(displacements_lqg[:step, midpoint_vertical_index])
    #     print(f"[LQG] Avg midpoint disp up to step {step}: {disp_mean:.2e}")
        
        
        
# %% MLP Training
# Prepare training data
inputs_raw = torch.tensor([[d, v, a] for d, v, a, _ in lqg_training_data], dtype=torch.float32)
targets_raw = torch.tensor([[f] for _, _, _, f in lqg_training_data], dtype=torch.float32)

# Normalize inputs
inputs_mean = inputs_raw.mean(0)
inputs_std = inputs_raw.std(0) + 1e-8  # avoid divide-by-zero
inputs_norm = (inputs_raw - inputs_mean) / inputs_std

# Normalize targets
targets_mean = targets_raw.mean(0)
targets_std = targets_raw.std(0) + 1e-8
targets_norm = (targets_raw - targets_mean) / targets_std

# Define MLP model
class MLPController(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=1):
        super(MLPController, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

mlp_controller = MLPController()

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(mlp_controller.parameters(), lr=1e-3)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
epochs = 2000
loss_history = []

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = mlp_controller(inputs_norm)
    loss = criterion(predictions, targets_norm)
    loss.backward()
    optimizer.step()
    # scheduler.step()

    loss_history.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")
        
# %% Simulation Loop (MLP)
mlp_forces = np.zeros(n_steps)
for step in range(1, n_steps):
    t = step * dt
    F = np.zeros(total_DOF)

    # Apply impact force over the specified duration
    if impact_time <= t < impact_time + impact_duration:
        F[midpoint_vertical_index] = impact_magnitude

    # Clone the base force vector for each controller to isolate their actions
    # F_uncontrolled_base = np.copy(F)
    # F_pd_base = np.copy(F)
    F_mlp_base = np.copy(F)
    
    # ---------------------- MLP Control ----------------------
    # F_mlp = M @ ((1 / (beta_n * dt**2)) * displacements_mlp[step - 1] +
    #                       (1 / (beta_n * dt)) * velocities_mlp[step - 1] +
    #                       ((0.5 / beta_n) - 1) * np.zeros(total_DOF)) + \
    #                   C @ ((gamma / (beta_n * dt)) * displacements_mlp[step - 1] +
    #                       (gamma / beta_n - 1) * velocities_mlp[step - 1] +
    #                       dt * ((gamma / (2 * beta_n)) - 1) * np.zeros(total_DOF))
        

    # # Solve for uncontrolled displacements and velocities
    # displacements_mlp_new = K_inv @ F_mlp
    # velocities_mlp_new = (displacements_mlp_new - displacements_mlp[step - 1]) / dt
    # accelerations_mlp[step - 1, :] = np.linalg.solve(
    #     M, F_mlp - C @ velocities_mlp[step - 1] - K @ displacements_mlp[step - 1])

    # displacements_mlp[step, :] = displacements_mlp_new
    # velocities_mlp[step, :] = velocities_mlp_new
    
    # MLP control force calculation
    state_input = torch.tensor([
        displacements_mlp[step - 1, control_vertical_index],
        velocities_mlp[step - 1, control_vertical_index],
        accelerations_mlp[step - 1, control_vertical_index]
    ], dtype=torch.float32)
    
    if t < impact_time:
        control_force_mlp = 0.0
    else:
        # Normal prediction logic
        state_input_norm = (state_input - inputs_mean) / inputs_std
        control_force_mlp = mlp_controller(state_input_norm).item()
        control_force_mlp = control_force_mlp * targets_std.item() + targets_mean.item()
        control_force_mlp = np.clip(control_force_mlp, -MAX_FORCE, MAX_FORCE)

    
    # Piezoelectric control forces at adjacent nodes
    # F_mlp_base[axial_index_1] += control_force_mlp   # +N
    # F_mlp_base[axial_index_2] -= control_force_mlp   # -N
    
    control_moment_mlp = moment_scaling_factor * control_force_mlp * (thickness / 2)
    F_mlp_base[rotation_index_1] += control_moment_mlp
    F_mlp_base[rotation_index_2] -= control_moment_mlp 
    
    if t >= impact_time:
        assert np.any(F_mlp_base != 0), f"[MLP] Step {step} has zero force vector!"

    # Solve for the new displacement and velocity for the system with the MLP control
    F_mlp_eff = F_mlp_base + M @ ((1 / (beta_n * dt**2)) * displacements_mlp[step - 1] +
                         (1 / (beta_n * dt)) * velocities_mlp[step - 1] +
                         ((0.5 / beta_n) - 1) * np.zeros(total_DOF)) + \
                C @ ((gamma / (beta_n * dt)) * displacements_mlp[step - 1] +
                     (gamma / beta_n - 1) * velocities_mlp[step - 1] +
                     dt * ((gamma / (2 * beta_n)) - 1) * np.zeros(total_DOF))

    # Solve for the new displacement and velocity
    displacements_mlp_new = K_inv @ F_mlp_eff
    velocities_mlp_new = (displacements_mlp_new - displacements_mlp[step - 1]) / dt

    # Store new state for MLP-controlled system
    accelerations_mlp[step, :] = np.linalg.solve(
        M, F_mlp_eff - C @ velocities_mlp_new - K @ displacements_mlp_new)
    displacements_mlp[step, :] = displacements_mlp_new
    velocities_mlp[step, :] = velocities_mlp_new
    displacement_history_mlp[step, :] = displacements_mlp_new
    mlp_forces[step] = control_force_mlp
    
    # if step % 1000 == 0:
    #     print(f"[MLP] Step {step}: input = {state_input.tolist()}, output force = {control_force_mlp:.4f}")

    
# %% Plot colors
cmap = plt.get_cmap("tab10")
# Select colors from the palette (indices 0-9) for tab10
# color_0 = cmap(0)  # Blue
# color_1 = cmap(1)  # Orange
# color_2 = cmap(2)  # Green
# color_3 = cmap(3)  # Red
# color_4 = cmap(4)  # Purple
# color_5 = cmap(5)  # Brown
# color_6 = cmap(6)  # Pink
# color_7 = cmap(7)  # Gray
# color_8 = cmap(8)  # Yellow
# color_9 = cmap(9)  # Cyan

tab20cmap = plt.get_cmap("tab20")
uncontcolor = cmap(7)
pdcolor = cmap(0)
mlpcolor = cmap(1)
lqgcolor = cmap(2)

# %% Visualization of vertical displacement and acceleration at selected nodes
# observation_nodes = [n_nodes // 8, n_nodes // 4, n_nodes // 2, 3 * n_nodes // 4, 7 * n_nodes // 8]
# node_labels = ["left", "1/4", "mid", "3/4", "right"]
# time_array = np.linspace(0, t_total, n_steps)
# plt.figure(figsize=(10, 4))
# for i, node in enumerate(observation_nodes):
#     idx = DOF_per_node * node + 1  # vertical DOF
#     plt.plot(time_array, displacement_history_uncontrolled[:, idx], label=f"disp {node_labels[i]}")
# plt.xlabel("time (s)")
# plt.ylabel("vertical displacement (m)")
# plt.title("Vertical Displacement at Multiple Nodes (Uncontrolled)")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(10, 4))
# for i, node in enumerate(observation_nodes):
#     idx = DOF_per_node * node + 1  # vertical DOF
#     plt.plot(time_array, accelerations_uncontrolled[:, idx], label=f"accel {node_labels[i]}")
# plt.xlabel("time (s)")
# plt.ylabel("vertical acceleration (m/s²)")
# plt.title("Vertical Acceleration at Multiple Nodes (Uncontrolled)")
# plt.grid(True)
# plt.legend()
# plt.xlim(0.075, 0.175)
# plt.tight_layout()
# plt.show()

# %% Plot beam displacements
plt.rc('font', family='Times New Roman', size=10)
plt.figure(figsize=(6.5, 3))
time_array = np.linspace(0, t_total, n_steps)

# Uncontrolled displacement at midpoint
plt.plot(time_array, displacement_history_uncontrolled[:, midpoint_vertical_index], label="uncontrolled", color=uncontcolor, linewidth=1.0)

# LQG controlled displacement at midpoint
# plt.plot(time_array, displacement_history_lqg[:, midpoint_vertical_index], label="LQG controlled", color=lqgcolor, linewidth=1.0)

# PD controlled displacement at midpoint
plt.plot(time_array, displacement_history_pd[:, midpoint_vertical_index], label="PD controlled", linewidth=1.0)#color=pdcolor,

# MLP controlled displacement at midpoint
plt.plot(time_array, displacement_history_mlp[:, midpoint_vertical_index], label="MLP controlled", linewidth=1.0)#color=mlpcolor,

plt.xlabel("time (s)")
plt.ylabel("displacement (m)")
plt.grid()
plt.legend(facecolor="white", edgecolor="lightgray", framealpha=1, frameon=True)
plt.show()

# %% Plot LQG vs MLP controlled beam displacements
plt.rc('font', family='Times New Roman', size=10)
plt.figure(figsize=(6.5, 3))
time_array = np.linspace(0, t_total, n_steps)

# MLP controlled displacement at midpoint
plt.plot(time_array, displacement_history_mlp[:, midpoint_vertical_index], label="MLP controlled", color=mlpcolor, linewidth=1.0)

# LQG controlled displacement at midpoint
plt.plot(time_array, displacement_history_lqg[:, midpoint_vertical_index], label="LQG controlled", color=lqgcolor, linewidth=1.0)

plt.xlabel("time (s)")
plt.ylabel("displacement (m)")
plt.grid()
plt.legend(facecolor="white", edgecolor="lightgray", framealpha=1, frameon=True)
plt.xlim(0.0, 1.5)
plt.show()

# %% Learning Curve Plotting
plt.figure()
plt.plot(loss_history)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()

# %% Control Force Plotting
plt.figure()
plt.plot(time_array, pd_forces, label="PD Force", color=pdcolor)
plt.plot(time_array, lqg_forces, label="LQG Force", color=lqgcolor)
plt.plot(time_array, mlp_forces, label="MLP force", color=mlpcolor)
plt.xlabel("time (s)")
plt.ylabel("Force (N)")
plt.grid()
plt.title("LQG Control Force Over Time")
plt.legend()
plt.show()