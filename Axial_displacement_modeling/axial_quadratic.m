
clear; clc; close all;

% modified Beam Formulation with Axial DOF
% nElem = 50;                 % number of elements
E = 210e9;                    % Young's modulus (Pa)
A = 0.01;                     % cross-sectional area (m^2)
I = 8.333e-6;                 % moment of inertia (m^4)
L = 2;                        % total beam length (m)

% moment or force assignment
force_node = round((nElem/2) + 1); % middle node (rotation DOF)
force_DOF_local = 3;              % 1 = u, 2 = w, 3 = theta
moment_value = 800;              % moment value (Nm)

% necessary parameters
nNode = nElem + 1;             % total number of main nodes
nMidNode = nElem;              % mid-nodes for axial only
nTotalNodes = nNode + nMidNode; % Total nodes including midpoints
nDOF_per_mainnode = 3;         % u, w, theta per main node
nDOF_per_midnode = 1;          % u only at mid-node
nDOF = nDOF_per_mainnode * nNode + nDOF_per_midnode * nMidNode; % Total DOFs
Le = L / nElem;                % Element length (m)

% atiffness matrix creation
% axial (quadratic) part: 3x3
ke_axial = (2*E*A/Le) * [7/6, -4/3, 1/6;
                        -4/3, 8/3, -4/3;
                         1/6, -4/3, 7/6];

% Bending (Hermite cubic) part: 4x4
ke_bending = (E*I/Le^3) * [12, 6*Le, -12, 6*Le;
                            6*Le, 4*Le^2, -6*Le, 2*Le^2;
                           -12, -6*Le, 12, -6*Le;
                            6*Le, 2*Le^2, -6*Le, 4*Le^2];

% --- ASSEMBLE GLOBAL STIFFNESS MATRIX ---
K = zeros(nDOF);

for e = 1:nElem
    % Global DOF indices for element
    left_node = e;
    right_node = e+1;
    mid_node = e;
    
    idx = [ ...
        (left_node-1)*3 + 1, (left_node-1)*3 + 2, (left_node-1)*3 + 3, ...
        3*nNode + mid_node, ...
        (right_node-1)*3 + 1, (right_node-1)*3 + 2, (right_node-1)*3 + 3 ];
    
    % local element stiffness matrix (7x7)
    ke = zeros(7,7);
    ke([1,4,5],[1,4,5]) = ke_axial;  % axial part
    ke([2,3,6,7],[2,3,6,7]) = ke_bending; % bending part

    % assemble to global
    K(idx,idx) = K(idx,idx) + ke;
end

% aSSEMBLE GLOBAL FORCE VECTOR 
F = zeros(nDOF,1);

% moment applied at rotation DOF of middle main node
globalDOF = (force_node-1)*3 + force_DOF_local;
F(globalDOF) = moment_value;

% applying bondary conditions
fixedDOF = [1,2,3, 3*nNode - 2, 3*nNode - 1, 3*nNode];
freeDOF = setdiff(1:nDOF, fixedDOF);

% Modify K and F
K_reduced = K(freeDOF,freeDOF);
F_reduced = F(freeDOF);

% reduced k and f
U_reduced = K_reduced \ F_reduced;

% displacement driving
U = zeros(nDOF,1);
U(freeDOF) = U_reduced;

% deflection plotting
x_main_nodes = linspace(0, L, nNode);  
vertical_disp = zeros(1, nNode);

for i = 1:nNode
    vertical_disp(i) = U((i-1)*3 + 2); % extract vertical displacement (w)
end

figure;
plot(x_main_nodes, vertical_disp, 'b-o', 'LineWidth', 2, 'MarkerFaceColor','r');
grid on;
xlabel('beam Length (m)');
ylabel('deflection (m)');
title('static deflected of extended Euler-Bernoulli beam (Quadratic Axial, Hermite Bending)');

