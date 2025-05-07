% Beam Formulation with Axial DOF, Mass Matrix, Modal Analysis, and Rayleigh Damping
clear; clc; close all;

%% input parameters
nElem = 50;              % number of elements
E     = 210e9;           % Young's modulus (Pa)
A     = 0.01;            % cross-sectional area (m^2)
I     = 8.333e-6;        % moment of inertia (m^4)
rho   = 7850;            % mass density (kg/m^3)
L     = 2;               % total beam length (m)
q     = 0;               % distributed load (ignored for static)

%% this is used for assigning a single static moment%%

%  mid-span moment assignment (for static test)
% force_node      = round((nElem/2) + 1);
% force_DOF_local = 3;     % 1=u, 2=w, 3=theta
% moment_value    = 800;   % applied moment (Nm)

%% derived sizes
nNode         = nElem + 1;
nDOF_per_node = 3;        % u, w, theta
nDOF          = nNode * nDOF_per_node;
Le            = L / nElem;

%% Element stiffness matrices
% axial (2×2)
ke_axial = (E*A/Le)*[1 -1; -1 1];

% bending (4×4 Hermite)
ke_bending = (E*I/Le^3)*[ ...
     12      6*Le   -12    6*Le; 
    6*Le   4*Le^2  -6*Le  2*Le^2;
   -12    -6*Le     12    -6*Le;
    6*Le   2*Le^2  -6*Le   4*Le^2 ];

% assemble into 6×6 element stiffness
ke = zeros(6,6);
axial_dofs   = [1 4];
bending_dofs = [2 3 5 6];
ke(axial_dofs,   axial_dofs)   = ke_axial;
ke(bending_dofs, bending_dofs) = ke_bending;

%% Element mass matrices (consistent)
% axial (2×2)
me_axial = (rho*A*Le/6)*[2 1; 1 2];

% bending (4×4 Hermite)
me_bending = (rho*A*Le/420)*[ ...
    156       22*Le    54      -13*Le;
     22*Le   4*Le^2   13*Le   -3*Le^2;
     54       13*Le   156     -22*Le;
    -13*Le   -3*Le^2  -22*Le   4*Le^2 ];

% assemble into 6×6 element mass
me = zeros(6,6);
me(axial_dofs,   axial_dofs)   = me_axial;
me(bending_dofs, bending_dofs) = me_bending;

%% Global assembly
K = zeros(nDOF);
M = zeros(nDOF);
for e = 1:nElem
    idx = (e-1)*nDOF_per_node + (1:6);
    K(idx,idx) = K(idx,idx) + ke;
    M(idx,idx) = M(idx,idx) + me;
end

%% Static loading: Apply Two Moments (Predefined)
F = zeros(nDOF,1);

% Moment 1
force_node_1      = 26;
force_DOF_local_1 = 3;   % rotation (theta)
moment_value_1    = -800; % Nm
globalDOF_1 = (force_node_1-1)*nDOF_per_node + force_DOF_local_1;
F(globalDOF_1) = moment_value_1;

%  Moment 2 
force_node_2      = 35;
force_DOF_local_2 = 3;    % rotation (theta)
moment_value_2    = +800; % Nm in opposite direction
globalDOF_2 = (force_node_2-1)*nDOF_per_node + force_DOF_local_2;
F(globalDOF_2) = moment_value_2;


%% Apply BCs (fixed at both ends: all DOFs)
fixedDOF = [1,2,3, nDOF-2, nDOF-1, nDOF];
freeDOF  = setdiff(1:nDOF, fixedDOF);

K_reduced = K(freeDOF, freeDOF);
M_reduced = M(freeDOF, freeDOF);
F_reduced = F(freeDOF);

%% Solve static problem
U = zeros(nDOF,1);
U(freeDOF) = K_reduced \ F_reduced;

% Plot static deflection
x_nodes = linspace(0, L, nNode);
vertical_disp = U(2:nDOF_per_node:end);
figure;
plot(x_nodes, vertical_disp, 'b-o','LineWidth',2,'MarkerFaceColor','r');
grid on;
xlabel('beam length (m)');
ylabel('vertical deflection (m)');
title('static deflection of beam with axial DOF');

%% Modal analysis
% Solve generalized eigenvalue problem: K_reduced * phi = omega^2 * M_reduced * phi
[Phi, D] = eig(K_reduced, M_reduced);
omega2 = diag(D);
[omega2, sortIdx] = sort(omega2);
Phi = Phi(:,sortIdx);
freq = sqrt(omega2)/(2*pi);

% Display first few natural frequencies
nModesToShow = 5;
disp('natural frequencies (Hz):');
disp(freq(1:nModesToShow));

% Plot first nModesToShow mode shapes (vertical deflection)
figure;
for j = 1:nModesToShow
    mode_full = zeros(nDOF,1);
    mode_full(freeDOF) = Phi(:,j);
    vmode = mode_full(2:nDOF_per_node:end);
    subplot(nModesToShow,1,j);
    plot(x_nodes, vmode, '-o','LineWidth',1.5);
    ylabel(sprintf('mode %d', j));
    grid on;
    if j==1
        title('mode shapes');
    end
    if j==nModesToShow
        xlabel('beam length (m)');
    end
end

%% Rayleigh damping calculation
%     damping ratios for first two modes
zeta1 = 0.02;  % 2% for mode 1
zeta2 = 0.02;  % 2% for mode 2
omega1 = 2*pi*freq(1);
omega2 = 2*pi*freq(2);

% deriving the alpha and beta: zeta_i = alpha/(2*omega_i) + beta*omega_i/2
Acoef = [1/(2*omega1), omega1/2; 1/(2*omega2), omega2/2];
bcoef = [zeta1; zeta2];
ab = Acoef \ bcoef;
alpha = ab(1);
beta  = ab(2);

% Assembled damping matrix
C_reduced = alpha * M_reduced + beta * K_reduced;

disp('Rayleigh damping coefficients:');
fprintf('  alpha = %.3e  beta = %.3e\n', alpha, beta);

% (Optional) Dynamic stiffness at a given frequency
% omega_test = 2*pi*10;  % 10 Hz
% Kdyn = -omega_test^2 * M_reduced + 1i*omega_test * C_reduced + K_reduced;


%% Newmark-Beta Time Integration (Impulse Response)

% parameters for time integration
beta   = 1/4;
gamma  = 1/2;
dt     = 1e-4;       
Tmax   = 0.1;        
Nt     = round(Tmax / dt);
time   = (0:Nt)*dt;

% initial conditions
w0  = zeros(length(freeDOF),1);  % initial displacement
v0  = zeros(length(freeDOF),1);  % initial velocity
a0  = M_reduced \ (F_reduced - C_reduced*v0 - K_reduced*w0);  % initial acceleration

% history arrays
W  = zeros(length(w0), Nt+1);  W(:,1) = w0;
V  = zeros(length(w0), Nt+1);  V(:,1) = v0;
A  = zeros(length(w0), Nt+1);  A(:,1) = a0;

% newmark coefficients
a0N = 1 / (beta*dt^2);
a1N = gamma / (beta*dt);
a2N = 1 / (beta*dt);
a3N = 1 / (2*beta) - 1;
a4N = gamma / beta - 1;
a5N = dt * (gamma / (2*beta) - 1);

% Effective stiffness matrix
Keff = K_reduced + a1N * C_reduced + a0N * M_reduced;

% impulse force imposing 
f_dyn = zeros(length(w0), Nt+1);
impulse_node = round(nElem/2) + 1;
impulse_DOF  = 2; % vertical DOF (w)
impulse_val  = 1000;   % N
impulse_time = 0.001;  % 1 ms duration
impulse_idx  = (impulse_node-1)*nDOF_per_node + impulse_DOF;
impulse_dof  = find(freeDOF == impulse_idx);  % local DOF in reduced system
impulse_steps = round(impulse_time / dt);
f_dyn(impulse_dof, 1:impulse_steps) = impulse_val;

% time integration main loop
for n = 1:Nt
    % effective force
    Feff = f_dyn(:,n+1) ...
         + M_reduced * (a0N*W(:,n) + a2N*V(:,n) + a3N*A(:,n)) ...
         + C_reduced * (a1N*W(:,n) + a4N*V(:,n) + a5N*A(:,n));

    % solve for displacement
    W(:,n+1) = Keff \ Feff;

    % acceleration and velocity updates
    A(:,n+1) = a0N * (W(:,n+1) - W(:,n)) - a2N * V(:,n) - a3N * A(:,n);
    V(:,n+1) = V(:,n) + dt * ((1 - gamma) * A(:,n) + gamma * A(:,n+1));
end

% plot dynamic response at midspan
mid_dof = find(freeDOF == impulse_idx);
figure;
plot(time, W(mid_dof,:), 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Vertical displacement at midpoint (m)');
title('Dynamic response to impulse force (Newmark-Beta)');
grid on;

% plot velocity at midpoint
figure;
plot(time, V(mid_dof,:), 'r', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Velocity at midpoint (m/s)');
title('Velocity response at midpoint (Newmark-Beta)');
grid on;

% plot acceleration at midpoint
figure;
plot(time, A(mid_dof,:), 'g', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Acceleration at midpoint (m/s^2)');
title('Acceleration response at midpoint (Newmark-Beta)');
grid on;

