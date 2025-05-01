
% including consistent mass matrix assembly and modal analysis

clear; clc; close all;

% input parameters
nElem = 50;              % number of elements
E     = 210e9;           % Young's modulus (Pa)
A     = 0.01;            % cross-sectional area (m^2)
I     = 8.333e-6;        % moment of inertia (m^4)
rho   = 7850;            % mass density (kg/m^3)
L     = 2;               % total beam length (m)

% moment or force assignment
force_node      = round((nElem/2) + 1); % mid-span node
force_DOF_local = 3;                    % 1=u, 2=w, 3=theta
moment_value    = 800;                  % applied moment (Nm)

%% derived sizes
nNode         = nElem + 1;
nDOF_per_node = 3;        % u, w, theta
nDOF          = nNode * nDOF_per_node;
Le            = L / nElem;

%% element stiffness
ke_axial = (E*A/Le)*[1 -1; -1 1];
ke_bending = (E*I/Le^3)*[ ...
     12      6*Le   -12    6*Le; 
    6*Le   4*Le^2  -6*Le  2*Le^2;
   -12    -6*Le     12    -6*Le;
    6*Le   2*Le^2  -6*Le   4*Le^2 ];
ke = zeros(6,6);
axial_dofs   = [1 4];
bending_dofs = [2 3 5 6];
ke(axial_dofs,   axial_dofs)   = ke_axial;
ke(bending_dofs, bending_dofs) = ke_bending;

%% element mass 
me_axial = (rho*A*Le/6)*[2 1; 1 2];
me_bending = (rho*A*Le/420)*[ ...
    156       22*Le    54      -13*Le;
     22*Le   4*Le^2   13*Le   -3*Le^2;
     54       13*Le   156     -22*Le;
    -13*Le   -3*Le^2  -22*Le   4*Le^2 ];
me = zeros(6,6);
me(axial_dofs,   axial_dofs)   = me_axial;
me(bending_dofs, bending_dofs) = me_bending;

%% global assembly
K = zeros(nDOF);
M = zeros(nDOF);
for e = 1:nElem
    idx = (e-1)*nDOF_per_node + (1:6);
    K(idx,idx) = K(idx,idx) + ke;
    M(idx,idx) = M(idx,idx) + me;
end

%% force vector
F = zeros(nDOF,1);
globalDOF = (force_node-1)*nDOF_per_node + force_DOF_local;
F(globalDOF) = moment_value;

%% apply boundary conditions (fixed at both ends: all DOFs)
fixedDOF = [1,2,3, nDOF-2, nDOF-1, nDOF];
freeDOF  = setdiff(1:nDOF, fixedDOF);

K_reduced = K(freeDOF, freeDOF);
M_reduced = M(freeDOF, freeDOF);
F_reduced = F(freeDOF);

%% solve static problem (stiffness only)
U = zeros(nDOF,1);
U(freeDOF) = K_reduced \ F_reduced;

%% plot static vertical deflection
x_nodes = linspace(0, L, nNode);
vertical_disp = U(2:nDOF_per_node:end);

figure;
plot(x_nodes, vertical_disp, 'b-o', 'LineWidth', 2, 'MarkerFaceColor','r');
grid on;
xlabel('Beam length (m)');
ylabel('Vertical deflection (m)');
title('Static deflection of modified Euler–Bernoulli beam');
legend('Deflection');

%% modal analysis
% generalized eigenvalue problem K x = omega^2 M x
[Modes, D] = eig(K_reduced, M_reduced);
omega2 = diag(D);
[omega2, sortIdx] = sort(omega2);
Modes = Modes(:,sortIdx);
freq = sqrt(omega2)/(2*pi);

%  first 5 natural frequencies
nModesToShow = 4;
disp('First 5 natural frequencies (Hz):');
disp(freq(1:nModesToShow));

% plot first 3 mode shapes
figure;
for j = 1:nModesToShow
    mode_full = zeros(nDOF,1);
    mode_full(freeDOF) = Modes(:,j);
    vmode = mode_full(2:nDOF_per_node:end);
    subplot(nModesToShow,1,j);
    plot(x_nodes, vmode, '-o','LineWidth',1.5);
    ylabel(sprintf('Mode %d',j));
    grid on;
    if j==1, title('First three vertical mode shapes'); end
    if j==3, xlabel('Beam length (m)'); end
end
