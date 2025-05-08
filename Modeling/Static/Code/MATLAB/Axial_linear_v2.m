% modified Beam Formulation with Axial DOF (fixed–fixed under mid-span moment)

clear; clc; close all;

%% input parameters
nElem = 50;              % number of elements
E = 210e9;               % Young's modulus (Pa)
A= 0.01;                 % cross-sectional area (m^2)
I= 8.333e-6;             % moment of inertia (m^4)
L= 2;                    % total beam length (m)

% moment or force assignment
force_node = round((nElem/2) + 1); % middle node (rotation DOF)
force_DOF_local = 3;              % 1 = u, 2 = w, 3 = theta
moment_value = 800;              % moment value (Nm)

%% derived sizes
nNode         = nElem + 1;            % total nodes
nDOF_per_node = 3;                    % u, w, θ per node
nDOF          = nNode * nDOF_per_node;
Le       = L / nElem;            % element length

%% element stiffness
% axial (2×2)
ke_axial = (E*A/Le) * [1 -1; -1 1];

% bending (4×4 Hermite)
ke_bending = (E*I/Le^3) * [...
     12     6*Le   -12    6*Le;
    6*Le   4*Le^2 -6*Le  2*Le^2;
    -12   -6*Le     12   -6*Le;
    6*Le   2*Le^2 -6*Le  4*Le^2];

% assemble into 6×6 with correct DOF mapping
ke = zeros(6,6);
axial_dofs   = [1 4];         % u₁, u₂
bending_dofs = [2 3 5 6];     % w₁, θ₁, w₂, θ₂

ke(axial_dofs,   axial_dofs)   = ke_axial;
ke(bending_dofs, bending_dofs) = ke_bending;

%% global assembly
K = zeros(nDOF);
for e = 1:nElem
    idx = (e-1)*nDOF_per_node + (1:6);
    K(idx,idx) = K(idx,idx) + ke;
end

%% force vector
F = zeros(nDOF,1);
globalDOF = (force_node-1)*nDOF_per_node + force_DOF_local;
F(globalDOF) = moment_value;

%% apply boundary conditions (fixed at both ends: all DOFs)
fixedDOF = [1,2,3, nDOF-2, nDOF-1, nDOF];
freeDOF  = setdiff(1:nDOF, fixedDOF);

K_reduced = K(freeDOF, freeDOF);
F_reduced = F(freeDOF);

%% solve
U = zeros(nDOF,1);
U(freeDOF) = K_reduced \ F_reduced;

%% plot vertical deflection
x_nodes = linspace(0, L, nNode);
vertical_disp = U(2:nDOF_per_node:end);

figure;
plot(x_nodes, vertical_disp, 'b-o', 'LineWidth', 2, 'MarkerFaceColor','r');
grid on;
xlabel('beam length (m)');
ylabel('vertical deflection (m)');
title('static deflected shape of modified Euler-Bernoulli beam');
legend('static deflection');
