clear; clc; close all;
% Directory to save figures
saveDir = 'C:\Users\trott\Dropbox\Conference Papers\In Progress\Roberts2025_IMECE\Latex\Figures';
if ~exist(saveDir, 'dir')
    mkdir(saveDir);
end

%% FEM Model Setup
% nElem = 49; E = 210e9; A = 0.01; I = 8.333e-6; rho = 7850; L = 2;
%nElem = 49; E = 210e9; width = 0.05; thickness = 0.005; rho = 7850; L = 1;
nElem = 49; E = 1.8602e10; width = 0.0254; thickness = 0.0016002; rho = 515.379; L = 0.0889;
A = width * thickness; I = (width * thickness^3) / 12; 
nNode = nElem + 1; dof_per_node = 3; totalDOF = nNode * dof_per_node;
Le = L / nElem;
nFrames = 50; colors = lines(nFrames);

% Element matrices
ke_axial = (E*A/Le)*[1 -1; -1 1];
ke_bend = (E*I/Le^3)*[12 6*Le -12 6*Le; 6*Le 4*Le^2 -6*Le 2*Le^2;
                      -12 -6*Le 12 -6*Le; 6*Le 2*Le^2 -6*Le 4*Le^2];
ke = zeros(6); me = zeros(6);
axial = [1 4]; bend = [2 3 5 6];
ke(axial,axial) = ke_axial; ke(bend,bend) = ke_bend;

me_axial = (rho*A*Le/6)*[2 1; 1 2];
me_bend = (rho*A*Le/420)*[156 22*Le 54 -13*Le; 22*Le 4*Le^2 13*Le -3*Le^2;
                          54 13*Le 156 -22*Le; -13*Le -3*Le^2 -22*Le 4*Le^2];
me(axial,axial) = me_axial; me(bend,bend) = me_bend;

K = zeros(totalDOF); M = zeros(totalDOF);
for e = 1:nElem
    idx = (e-1)*dof_per_node + (1:6);
    K(idx,idx) = K(idx,idx) + ke;
    M(idx,idx) = M(idx,idx) + me;
end

% Fixed-fixed boundary
fixedDOF = [1,2,3, totalDOF-2, totalDOF-1, totalDOF];
freeDOF = setdiff(1:totalDOF, fixedDOF);
Kf = K(freeDOF,freeDOF); Mf = M(freeDOF,freeDOF);

%% Static Deflection Under Moment Couple
% Prepare load vector
F_static = zeros(totalDOF, 1);

% Define same moment couple as in control: Nodes 20 and 30
nodeL = 18; nodeR = 32;
DOF_thetaL = (nodeL - 1)*dof_per_node + 3;
DOF_thetaR = (nodeR - 1)*dof_per_node + 3;

M0 = 0.5;  % Nm magnitude of static moment
F_static(DOF_thetaL) = -M0;
F_static(DOF_thetaR) = +M0;

% Apply boundary conditions
F_static_reduced = F_static(freeDOF);

% Solve static equilibrium: Kf * u = F_static
U_static = zeros(totalDOF, 1);
U_static(freeDOF) = Kf \ F_static_reduced;

% Extract vertical displacements
x_nodes = 1:nNode;
w_static = U_static(2:dof_per_node:end);  % vertical DOFs

figure('Units','inches','Position',[1 1 3.6 2.0]);
plot(x_nodes, 1e3*w_static, '-', 'LineWidth', 1);
hold on;

% Mark moment application nodes with red circles
plot(x_nodes([nodeL, nodeR]), 1e3*w_static([nodeL, nodeR]), 'o', ...
     'MarkerSize', 6, 'MarkerFaceColor', colors(2,:), 'Color', colors(2,:));

xlabel('node number', 'FontName', 'Times New Roman', 'FontSize', 11);
ylabel('vertical deflection (mm)', 'FontName', 'Times New Roman', 'FontSize', 11);
ylim([-0.7 0]);
grid on; box on;
print(gcf, fullfile(saveDir, 'static_deflection_moment1'), '-dpng', '-r300');


F_static(DOF_thetaL) = +M0;  % flipped
F_static(DOF_thetaR) = -M0;

% Apply BCs and solve
F_static_reduced = F_static(freeDOF);
U_static = zeros(totalDOF, 1);
U_static(freeDOF) = Kf \ F_static_reduced;

% Extract and plot
x_nodes = 1:nNode;
w_static = U_static(2:dof_per_node:end);  % vertical DOFs

figure('Units','inches','Position',[1 1 3.6 2.0]);
plot(x_nodes, 1e3*w_static, '-', 'LineWidth', 1);
hold on;

% Mark moment application nodes with red circles
plot(x_nodes([nodeL, nodeR]), 1e3*w_static([nodeL, nodeR]), 'o', ...
     'MarkerSize', 6, 'MarkerFaceColor', colors(2,:), 'Color', colors(2,:));
xlabel('node number', 'FontName', 'Times New Roman', 'FontSize', 11);
ylabel('vertical deflection (mm)', 'FontName', 'Times New Roman', 'FontSize', 11);
ylim([0 0.7]);
grid on; box on;
print(gcf, fullfile(saveDir, 'static_deflection_moment2'), '-dpng', '-r300');

%% Rayleigh damping
zeta1 = 0.02; zeta2 = 0.02;
[Phi,D] = eig(Kf,Mf); omega = sqrt(diag(D));
A = [1/(2*omega(1)) omega(1)/2; 1/(2*omega(2)) omega(2)/2];
ab = A\[zeta1; zeta2]; alpha = ab(1); beta = ab(2);
Cf = alpha*Mf + beta*Kf;

%% Time Integration Setup
dt = 1e-4; Tmax = 0.05; Nt = round(Tmax/dt); t = (0:Nt)*dt;
nDOF_f = length(freeDOF);

% Initial Conditions
W_free = zeros(nDOF_f,Nt+1); V = zeros(nDOF_f,Nt+1); Aacc = zeros(nDOF_f,Nt+1);

% Apply impulse near control region
impulse_node = round(25);  
mid_dof_global = (impulse_node-1)*dof_per_node + 2;
mid_dof = find(freeDOF==mid_dof_global);

f_dyn = zeros(nDOF_f, Nt+1);
impulse_val = 1000;

% Apply impulse starting at 0.1 ms (0.0001 s)
impulse_start_time = 0.005;  % seconds
impulse_duration = 0.001;     % impulse lasts for 1 ms
start_idx = round(impulse_start_time / dt);
num_impulse_steps = round(impulse_duration / dt);

% Apply impulse to f_dyn starting at start_idx
f_dyn(mid_dof, start_idx : start_idx + num_impulse_steps - 1) = impulse_val;

% Newmark coefficients
b = 1/4; g = 1/2;
a0 = 1/(b*dt^2); a1 = g/(b*dt); a2 = 1/(b*dt); a3 = 1/(2*b)-1;
a4 = g/b - 1; a5 = dt*(g/(2*b)-1);
Keff = Kf + a1*Cf + a0*Mf;

%% Free Response Simulation
W = W_free; Aacc(:,1) = Mf\(f_dyn(:,1) - Cf*V(:,1) - Kf*W(:,1));
for n = 1:Nt
    Feff = f_dyn(:,n+1) + ...
        Mf*(a0*W(:,n) + a2*V(:,n) + a3*Aacc(:,n)) + ...
        Cf*(a1*W(:,n) + a4*V(:,n) + a5*Aacc(:,n));
    W(:,n+1) = Keff \ Feff;
    Aacc(:,n+1) = a0*(W(:,n+1)-W(:,n)) - a2*V(:,n) - a3*Aacc(:,n);
    V(:,n+1) = V(:,n) + dt*((1-g)*Aacc(:,n) + g*Aacc(:,n+1));
end
W_free = W; A_free = Aacc; V_free = V;

% Control nodes (adjacent rotation DOFs)
left_node = 20; right_node = 30;
thetaL_global = (left_node - 1)*dof_per_node + 3;
thetaR_global = (right_node - 1)*dof_per_node + 3;
idL = find(freeDOF==thetaL_global); idR = find(freeDOF==thetaR_global);

if isempty(idL) || isempty(idR)
    error('Control node DOFs not found in freeDOF!');
end

%%
Kp_vals = logspace(-6, -1, 10);  % 10 values from 1e-6 to 1e-1
Kd_vals = logspace(-5, -1, 10);  % 10 values from 1e-5 to 1e-1
results = [];

fprintf('Running PD Tuning Sweep:\n');
for kp = Kp_vals
    for kd = Kd_vals
        fprintf('  Testing Kp = %.1e, Kd = %.1e ...\n', kp, kd);
        [peak, settle, rms_a] = run_PD_simulation(kp, kd, ...
            zeros(nDOF_f,Nt+1), zeros(nDOF_f,Nt+1), zeros(nDOF_f,Nt+1), ...
            f_dyn, Mf, Cf, Kf, Keff, dt, Nt, mid_dof, freeDOF, totalDOF, ...
            dof_per_node, left_node, right_node);

        results = [results; kp, kd, 1e3*peak, settle, rms_a];
    end
end

T = array2table(results, ...
    'VariableNames', {'Kp', 'Kd', 'PeakDisp_mm', 'SettleTime_ms', 'RMSAccel'});
disp(T);
sorted_by_peak = sortrows(T, 'PeakDisp_mm');
sorted_by_settle = sortrows(T, 'SettleTime_ms');
sorted_by_rms = sortrows(T, 'RMSAccel');
best = sorted_by_settle(1, :); 

%% PD Control Simulation
% Kp = 250; 
% Kd = 1000;
% Kp = 10; 
% Kd = 0.175;
% Kp = 0.00001; 
% Kd = 0.00005;
Kp = best.Kp;
Kd = best.Kd;
fprintf('Using optimal PD gains: Kp = %.2e, Kd = %.2e\n', Kp, Kd);

W_pd = zeros(nDOF_f,Nt+1); V_pd = zeros(nDOF_f,Nt+1); A_pd = zeros(nDOF_f,Nt+1);
f_pd = f_dyn; M_record = zeros(1,Nt);

for n = 1:Nt
    thetaL = W_pd(idL, n); thetaR = W_pd(idR, n);
    dthetaL = V_pd(idL, n); dthetaR = V_pd(idR, n);
    
    curvature = thetaR - thetaL;
    curvature_rate = dthetaR - dthetaL;
    M_control = -Kp * curvature - Kd * curvature_rate;
    
    % Store control history
    M_record(n) = M_control;

    % Apply control moment couple
    f_pd(idL, n+1) = f_pd(idL, n+1) - M_control;
    f_pd(idR, n+1) = f_pd(idR, n+1) + M_control;

    Feff = f_pd(:,n+1) + ...
        Mf*(a0*W_pd(:,n) + a2*V_pd(:,n) + a3*A_pd(:,n)) + ...
        Cf*(a1*W_pd(:,n) + a4*V_pd(:,n) + a5*A_pd(:,n));
    W_pd(:,n+1) = Keff \ Feff;
    A_pd(:,n+1) = a0*(W_pd(:,n+1) - W_pd(:,n)) - a2*V_pd(:,n) - a3*A_pd(:,n);
    V_pd(:,n+1) = V_pd(:,n) + dt*((1-g)*A_pd(:,n) + g*A_pd(:,n+1));
end

%% Plot Displacement Comparison
figure('Units','inches','Position',[1 1 3.6 2.0]);  % width x height in inches
plot(1e3 * t, 1e3 * W_free(mid_dof,:), 'LineWidth', 1); hold on;
plot(1e3 * t, 1e3 * W_pd(mid_dof,:), 'LineWidth', 1);
xlabel('time (ms)', 'FontName', 'Times New Roman', 'FontSize', 11);
ylabel('midpoint displacement (mm)', 'FontName', 'Times New Roman', 'FontSize', 11);
legend({'free', 'pd-controlled'}, 'FontName', 'Times New Roman', 'FontSize', 9, 'Location', 'northeast');
%title('free vs pd-controlled midpoint response', 'FontName', 'Times New Roman', 'FontSize', 11);
% xlim([0 1500]);
set(gca,'LooseInset',get(gca,'TightInset'))
grid on;
box on;
print(gcf, fullfile(saveDir, 'displacement_comparison'), '-dpng', '-r300');

%% Plot Control Moment History
figure('Units','inches','Position',[1 1 3.6 2.0]);
plot(1e3 * t(1:Nt), M_record, 'LineWidth', 1);
xlabel('time (ms)', 'FontName', 'Times New Roman', 'FontSize', 11);
ylabel('control moment (Nm)', 'FontName', 'Times New Roman', 'FontSize', 11);
%title('control moment applied over time', 'FontName', 'Times New Roman', 'FontSize', 11);
set(gca,'LooseInset',get(gca,'TightInset'))
grid on;
box on;
print(gcf, fullfile(saveDir, 'control_moment_history'), '-dpng', '-r300');

%% Reconstruct full DOF history
% U_hist = zeros(totalDOF, Nt+1);
% U_hist(freeDOF, :) = W_free;
% 
% % Extract vertical displacements
% w_hist = U_hist(2:dof_per_node:end, :);
% x_nodes = 1:nNode;
% t_hist = t;
% 
% frame_idx = round(linspace(1, size(w_hist, 2), nFrames));
% 
% % Create figure with fixed width
% figure('Units','inches','Position',[1 1 3.6 3.0]);  % taller to fit legend
% hold on;
% 
% for i = 1:nFrames
%     plot(x_nodes, 1e3 * w_hist(:, frame_idx(i)), '-', ...
%          'LineWidth', 1.2, ...
%          'Color', colors(i,:), ...
%          'DisplayName', sprintf('t = %.1f ms', 1e3 * t_hist(frame_idx(i))));
% end
% 
% xlabel('node number', 'FontName', 'Times New Roman', 'FontSize', 11);
% ylabel('vertical deflection (mm)', 'FontName', 'Times New Roman', 'FontSize', 11);
% grid on; box on;
% 
% % Move legend below plot, in multiple columns
% lgd = legend('Orientation','horizontal', ...
%              'NumColumns', 3, ...
%              'Location','southoutside', ...
%              'FontSize', 8, ...
%              'FontName', 'Times New Roman');
% 
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 10, 'LineWidth', 1);
% 
% % Tight layout for figure export
% set(gca, 'LooseInset', [0.03 0.03 0.03 0.03]);
% 
% % Save
% print(gcf, fullfile(saveDir, 'beam_shape_over_time_free'), '-dpng', '-r300');

%% Reconstruct full DOF history
U_hist = zeros(totalDOF, Nt+1);
U_hist(freeDOF, :) = W_free;

% Extract vertical displacements
w_hist = U_hist(2:dof_per_node:end, :);
x_nodes = 1:nNode;
t_hist = t;

% Choose frames
frame_idx = round(linspace(1, size(w_hist, 2), nFrames));

% Build colormap based on time progression
cmap = turbo(nFrames);  % Or use 'parula', 'jet', etc.

% Create figure
figure('Units','inches','Position',[1 1 3.6 3.0]);
hold on;

for i = 1:nFrames
    color = cmap(i, :);
    plot(x_nodes, 1e3 * w_hist(:, frame_idx(i)), '-', ...
         'LineWidth', 1.2, 'Color', color);
end

xlabel('node number', 'FontName', 'Times New Roman', 'FontSize', 11);
ylabel('vertical deflection (mm)', 'FontName', 'Times New Roman', 'FontSize', 11);
ylim([-15 30]);
grid on; box on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 11, 'LineWidth', 1);

% Add custom colorbar as time legend
colormap(cmap);
cb = colorbar('southoutside');
cb.Ticks = linspace(0, 1, 5);
cb.TickLabels = arrayfun(@(i) sprintf('%.0f', 1e3 * t_hist(frame_idx(i))), ...
                         round(linspace(1, nFrames, 5)), 'UniformOutput', false);
cb.Label.String = 'time scale (ms)';
cb.Label.FontName = 'Times New Roman';
cb.Label.FontSize = 11;

% Save
print(gcf, fullfile(saveDir, 'beam_shape_over_time_free_colormap'), '-dpng', '-r300');


%% Reconstruct full DOF history

% U_hist = zeros(totalDOF, Nt+1);
% U_hist(freeDOF, :) = W_free;
% 
% % Extract vertical displacements (mm)
% w_hist = 1e3 * U_hist(2:dof_per_node:end, :);  % nNode × Nt+1
% x_nodes = 1:nNode;
% t_ms = t * 1e3;
% 
% % Create figure
% figure('Units','inches','Position',[1 1 5 3]);
% hold on;
% 
% % Create a colormap for mapping deflection magnitude
% nPoints = numel(w_hist);
% w_min = min(w_hist(:));
% w_max = max(w_hist(:));
% cmap = turbo(256);
% 
% % Plot each line colored by its local deflection magnitude
% for k = 1:Nt+1
%     z_vals = w_hist(:, k);
%     y_vals = t_ms(k) * ones(size(x_nodes));
%     x_vals = x_nodes;
% 
%     % Normalize this line's color range
%     norm_vals = round(1 + (z_vals - w_min) / (w_max - w_min) * 255);
%     norm_vals = max(min(norm_vals, 256), 1);  % clip to [1, 256]
% 
%     for i = 1:(nNode - 1)
%         % Segment between adjacent nodes
%         plot3(x_vals(i:i+1), y_vals(i:i+1), z_vals(i:i+1), '-', ...
%               'Color', cmap(norm_vals(i),:), 'LineWidth', 1);
%     end
% end
% 
% % Axes and labels
% xlabel('Node number', 'FontName', 'Times New Roman', 'FontSize', 12);
% ylabel('Time (ms)', 'FontName', 'Times New Roman', 'FontSize', 12);
% zlabel('Vertical deflection (mm)', 'FontName', 'Times New Roman', 'FontSize', 12);
% title('Beam shape evolution (deflection-colored)', 'FontName', 'Times New Roman', 'FontSize', 12);
% 
% view(45, 25);
% grid on; box on;
% 
% % Add colorbar mapped to Z (deflection)
% colormap(cmap);
% cb = colorbar;
% cb.Label.String = 'Deflection (mm)';
% cb.Label.FontName = 'Times New Roman';
% cb.Label.FontSize = 11;
% 
% % Set colorbar tick labels to actual deflection values
% cb.Ticks = linspace(0, 1, 5);
% cb.TickLabels = arrayfun(@(v) sprintf('%.1f', v), ...
%     linspace(w_min, w_max, 5), 'UniformOutput', false);
% 
% print(gcf, fullfile(saveDir, 'beam_shape_3D_lines_by_deflection'), '-dpng', '-r600');


%% Performance Metrics
disp('Performance Comparison:');

% Metrics
peak_free = max(abs(W_free(mid_dof,:)));
peak_pd = max(abs(W_pd(mid_dof,:)));
rms_window = t <= 0.015;
rms_a_free = rms(A_free(mid_dof, rms_window));
rms_a_pd   = rms(A_pd(mid_dof, rms_window));

% Settling thresholds
settle_thresh_free = 0.05 * peak_free;
settle_thresh_pd   = 0.05 * peak_pd;
sustain_steps = round(0.005 / dt);  % 5 ms

[~, peak_idx_free] = max(abs(W_free(mid_dof,:)));
[~, peak_idx_pd]   = max(abs(W_pd(mid_dof,:)));

settle_free_ms = NaN;
settle_pd_ms   = NaN;

% Settling time detection
for k = peak_idx_free : Nt - sustain_steps
    window = abs(W_free(mid_dof, k : k + sustain_steps));
    if all(window < settle_thresh_free)
        settle_free_ms = t(k) * 1e3;
        break;
    end
end

for k = peak_idx_pd : Nt - sustain_steps
    window = abs(W_pd(mid_dof, k : k + sustain_steps));
    if all(window < settle_thresh_pd)
        settle_pd_ms = t(k) * 1e3;
        break;
    end
end

% Compute improvements
cond_Keff = cond(Keff);
peak_improvement = (peak_free - peak_pd) / peak_free * 100;
rms_improvement = (rms_a_free - rms_a_pd) / rms_a_free * 100;
fprintf('DEBUG: RMS free = %.4f, RMS pd = %.4f\n', rms_a_free, rms_a_pd);

if ~isnan(settle_free_ms) && ~isnan(settle_pd_ms) && settle_free_ms > 0
    settle_improvement = (settle_free_ms - settle_pd_ms) / settle_free_ms * 100;
else
    settle_improvement = NaN;
end

% Display results
fprintf('Condition number of Keff: %.2e\n', cond_Keff);
fprintf('%-25s %-12s %-12s %-12s\n', 'Metric', 'Free', 'PD', 'Improvement');
fprintf('%-25s %-12.2f %-12.2f %-12.2f\n', 'Peak Displacement (mm)', ...
        1e3 * peak_free, 1e3 * peak_pd, peak_improvement);
if ~isnan(settle_free_ms) && ~isnan(settle_pd_ms)
    fprintf('%-25s %-12.2f %-12.2f %-12.2f\n', 'Settling Time (ms)', ...
            settle_free_ms, settle_pd_ms, settle_improvement);
else
    fprintf('%-25s %-12s %-12s %-12s\n', 'Settling Time (ms)', ...
            settle_free_ms_str(settle_free_ms), ...
            settle_free_ms_str(settle_pd_ms), '--');
end
fprintf('%-25s %-12.2f %-12.2f %-12.2f\n', 'RMS Accel (m/s^2)', ...
        rms_a_free, rms_a_pd, rms_improvement);

% Helper function to print -- for NaNs
function out = settle_free_ms_str(val)
    if isnan(val)
        out = '--';
    else
        out = sprintf('%.2f', val);
    end
end

function [peak_pd, settle_pd_ms, rms_pd] = run_PD_simulation(Kp, Kd, ...
    W0, V0, A0, f_dyn, Mf, Cf, Kf, Keff, dt, Nt, mid_dof, freeDOF, totalDOF, dof_per_node, left_node, right_node)

    % Setup arrays
    nDOF_f = size(W0, 1);
    W = W0; V = V0; A = A0;
    f_pd = f_dyn;
    
    thetaL_global = (left_node - 1)*dof_per_node + 3;
    thetaR_global = (right_node - 1)*dof_per_node + 3;
    idL = find(freeDOF == thetaL_global); idR = find(freeDOF == thetaR_global);
    
    % Newmark coefficients
    b = 1/4; g = 1/2;
    a0 = 1/(b*dt^2); a1 = g/(b*dt); a2 = 1/(b*dt); a3 = 1/(2*b)-1;
    a4 = g/b - 1; a5 = dt*(g/(2*b)-1);
    
    % Time integration with PD control
    for n = 1:Nt
        thetaL = W(idL, n); thetaR = W(idR, n);
        dthetaL = V(idL, n); dthetaR = V(idR, n);

        curvature = thetaR - thetaL;
        curvature_rate = dthetaR - dthetaL;
        M_control = -Kp * curvature - Kd * curvature_rate;

        % Apply control moment couple
        f_pd(idL, n+1) = f_pd(idL, n+1) - M_control;
        f_pd(idR, n+1) = f_pd(idR, n+1) + M_control;

        Feff = f_pd(:,n+1) + ...
            Mf*(a0*W(:,n) + a2*V(:,n) + a3*A(:,n)) + ...
            Cf*(a1*W(:,n) + a4*V(:,n) + a5*A(:,n));

        W(:,n+1) = Keff \ Feff;
        A(:,n+1) = a0*(W(:,n+1)-W(:,n)) - a2*V(:,n) - a3*A(:,n);
        V(:,n+1) = V(:,n) + dt*((1-g)*A(:,n) + g*A(:,n+1));
    end

    % Metrics
    response = W(mid_dof, :);
    peak_pd = max(abs(response));
    rms_pd = rms(A(mid_dof, :));

    % Settling time
    settle_thresh = 0.05 * peak_pd;
    sustain_steps = round(0.005 / dt);
    [~, peak_idx] = max(abs(response));
    settle_pd_ms = NaN;

    for k = peak_idx:Nt - sustain_steps
        if all(abs(response(k:k + sustain_steps)) < settle_thresh)
            settle_pd_ms = (k * dt) * 1e3;
            break;
        end
    end
end
