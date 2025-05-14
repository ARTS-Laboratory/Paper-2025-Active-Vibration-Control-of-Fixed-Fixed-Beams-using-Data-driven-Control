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
nodeL = 16; nodeR = 34;
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
ylim([-0.8 0]);
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
ylim([0 0.8]);
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
left_node = 18; right_node = 32;
thetaL_global = (left_node - 1)*dof_per_node + 3;
thetaR_global = (right_node - 1)*dof_per_node + 3;
idL = find(freeDOF==thetaL_global); idR = find(freeDOF==thetaR_global);

if isempty(idL) || isempty(idR)
    error('Control node DOFs not found in freeDOF!');
end

%% PID Tuning Sweep
% Kp_vals = logspace(-2, -1, 10);
% Kd_vals = logspace(-6, -4, 10);
% Ki_vals = logspace(-8, -1, 5);
% results_pid = [];
% 
% peak_response = max(abs(W_free(mid_dof,:)));
% 
% fprintf('Running PID Tuning Sweep:\n');
% for kp = Kp_vals
%     for kd = Kd_vals
%         for ki = Ki_vals
%             fprintf('  Kp=%.1e, Kd=%.1e, Ki=%.1e ...\n', kp, kd, ki);
%             [peak, settle, rms_a] = run_PID_simulation(kp, kd, ki, ...
%                     zeros(nDOF_f,Nt+1), zeros(nDOF_f,Nt+1), zeros(nDOF_f,Nt+1), ...
%                     f_dyn, Mf, Cf, Kf, Keff, dt, Nt, t, mid_dof, freeDOF, totalDOF, ...
%                     dof_per_node, left_node, right_node, peak_response);
% 
%             results_pid = [results_pid; kp, kd, ki, 1e3*peak, settle, rms_a];
%         end
%     end
% end
% 
% T_pid = array2table(results_pid, ...
%     'VariableNames', {'Kp', 'Kd', 'Ki', 'PeakDisp_mm', 'SettleTime_ms', 'RMSAccel'});
% disp(T_pid);
% sorted_pid = sortrows(T_pid, {'PeakDisp_mm', 'SettleTime_ms', 'RMSAccel'});
% best_pid = sorted_pid(1,:);

%% Use Optimal PID Gains
% Kp = best_pid.Kp;
% Kd = best_pid.Kd;
% Ki = best_pid.Ki;
Kp = 0.117;
Kd = 0.000075;      
Ki = 0.01;
fprintf('Using optimal PID gains: Kp = %.2e, Kd = %.2e, Ki = %.2e\n', Kp, Kd, Ki);

%% Run PID Simulation for Plotting (with shutoff)
W_pid = zeros(nDOF_f,Nt+1); 
V_pid = zeros(nDOF_f,Nt+1); 
A_pid = zeros(nDOF_f,Nt+1);
f_pid = f_dyn; 
M_record_pid = zeros(1,Nt);
curvature_integral = 0;

peak_response = max(abs(W_free(mid_dof,:)));
control_threshold = 0.075 * peak_response;   
sustain_steps = round(0.005 / dt);          
below_thresh_counter = 0;
control_active = true;
decay_factor = 1.0;
decay_tau = 0.005;  
decay_started = false;

for n = 1:Nt
    % Check midpoint DOF magnitude
    if abs(W_pid(mid_dof, n)) < control_threshold
        below_thresh_counter = below_thresh_counter + 1;
    else
        below_thresh_counter = 0;
    end

    % Trigger decay if sustained below threshold
    if control_active && below_thresh_counter >= sustain_steps
        control_active = false;
        decay_started = true;
        decay_start_time = t(n);
    end

    % Update decay factor if in decay mode
    if decay_started
        elapsed_decay = t(n) - decay_start_time;
        decay_factor = exp(-elapsed_decay / decay_tau);
        % Optional: zero control if it’s negligible
        if decay_factor < 1e-4
            decay_factor = 0;
        end
    else
        decay_factor = 1.0;
    end

    % Control moment calculation (decay applied to full law)
    thetaL = W_pid(idL, n); thetaR = W_pid(idR, n);
    dthetaL = V_pid(idL, n); dthetaR = V_pid(idR, n);

    curvature = thetaR - thetaL;
    curvature_rate = dthetaR - dthetaL;
    curvature_integral = curvature_integral + curvature * dt;

    M_control = decay_factor * (-Kp * curvature - Kd * curvature_rate - Ki * curvature_integral);

    % Apply moment couple
    f_pid(idL, n+1) = f_pid(idL, n+1) - M_control;
    f_pid(idR, n+1) = f_pid(idR, n+1) + M_control;

    M_record_pid(n) = M_control;

    % Newmark Integration
    Feff = f_pid(:,n+1) + ...
        Mf*(a0*W_pid(:,n) + a2*V_pid(:,n) + a3*A_pid(:,n)) + ...
        Cf*(a1*W_pid(:,n) + a4*V_pid(:,n) + a5*A_pid(:,n));

    W_pid(:,n+1) = Keff \ Feff;
    A_pid(:,n+1) = a0*(W_pid(:,n+1)-W_pid(:,n)) - a2*V_pid(:,n) - a3*A_pid(:,n);
    V_pid(:,n+1) = V_pid(:,n) + dt*((1-g)*A_pid(:,n) + g*A_pid(:,n+1));
end

%% Plot Displacement Comparison with PID
figure('Units','inches','Position',[1 1 3.6 2.0]);
plot(1e3 * t, 1e3 * W_free(mid_dof,:), 'LineWidth', 1); hold on;
plot(1e3 * t, 1e3 * W_pid(mid_dof,:), 'LineWidth', 1);
xlabel('time (ms)', 'FontName', 'Times New Roman', 'FontSize', 11);
ylabel('midpoint displacement (mm)', 'FontName', 'Times New Roman', 'FontSize', 11);
legend({'uncontrolled', 'PID-controlled'}, 'FontName', 'Times New Roman', 'FontSize', 9);
ylim([-25 45]);
grid on; box on;
print(gcf, fullfile(saveDir, 'displacement_comparison_pid'), '-dpng', '-r300');

%% Plot PID Control Moment
figure('Units','inches','Position',[1 1 3.6 2.0]);
plot(1e3 * t(1:Nt), M_record_pid, 'LineWidth', 1);
xlabel('time (ms)', 'FontName', 'Times New Roman', 'FontSize', 11);
ylabel('control moment (Nm)', 'FontName', 'Times New Roman', 'FontSize', 11);
grid on; box on;
print(gcf, fullfile(saveDir, 'control_moment_pid'), '-dpng', '-r300');

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

%% Performance Metrics
disp('Performance Comparison:');

% Metrics
peak_free = max(abs(W_free(mid_dof,:)));
control_active = true;
control_threshold = 0.01 * peak_free;  % 1% of peak
sustain_steps = round(0.005 / dt);     % 5 ms
below_thresh_counter = 0;
peak_pid = max(abs(W_pid(mid_dof,:)));
rms_window = t <= 0.015;
rms_a_free = rms(A_free(mid_dof, rms_window));
rms_a_pid = rms(A_pid(mid_dof, rms_window));
rms_a_free_dB = 20 * log10(rms_a_free);
rms_a_pid_dB  = 20 * log10(rms_a_pid);

% Settling thresholds
settle_thresh_free = 0.05 * peak_free;
settle_thresh_pid   = 0.05 * peak_pid;
sustain_steps = round(0.005 / dt);  % 5 ms

[~, peak_idx_free] = max(abs(W_free(mid_dof,:)));
[~, peak_idx_pid]   = max(abs(W_pid(mid_dof,:)));

settle_free_ms = NaN;
settle_pid_ms   = NaN;

% Settling time detection
for k = peak_idx_free : Nt - sustain_steps
    window = abs(W_free(mid_dof, k : k + sustain_steps));
    if all(window < settle_thresh_free)
        settle_free_ms = t(k) * 1e3;
        break;
    end
end

for k = peak_idx_pid : Nt - sustain_steps
    window = abs(W_pid(mid_dof, k : k + sustain_steps));
    if all(window < settle_thresh_pid)
        settle_pid_ms = t(k) * 1e3;
        break;
    end
end

% Compute improvements
cond_Keff = cond(Keff);
peak_improvement = (peak_free - peak_pid) / peak_free * 100;
rms_improvement = (rms_a_free_dB - rms_a_pid_dB) / rms_a_free_dB * 100;
% fprintf('DEBUG: RMS free = %.4f, RMS pd = %.4f\n', rms_a_free, rms_a_pid);

if ~isnan(settle_free_ms) && ~isnan(settle_pid_ms) && settle_free_ms > 0
    settle_improvement = (settle_free_ms - settle_pid_ms) / settle_free_ms * 100;
else
    settle_improvement = NaN;
end

% Display results
fprintf('Condition number of Keff: %.2e\n', cond_Keff);
fprintf('%-25s %-12s %-12s %-12s\n', 'Metric', 'Free', 'PD', 'Improvement');
fprintf('%-25s %-12.2f %-12.2f %-12.2f\n', 'Peak Displacement (mm)', ...
        1e3 * peak_free, 1e3 * peak_pid, peak_improvement);
if ~isnan(settle_free_ms) && ~isnan(settle_pid_ms)
    fprintf('%-25s %-12.2f %-12.2f %-12.2f\n', 'Settling Time (ms)', ...
            settle_free_ms, settle_pid_ms, settle_improvement);
else
    fprintf('%-25s %-12s %-12s %-12s\n', 'Settling Time (ms)', ...
            settle_free_ms_str(settle_free_ms), ...
            settle_free_ms_str(settle_pid_ms), '--');
end
fprintf('%-25s %-12.2f %-12.2f %-12.2f\n', 'RMS Accel (dB)', ...
        rms_a_free_dB, rms_a_pid_dB, rms_improvement);

% Helper function to print -- for NaNs
function out = settle_free_ms_str(val)
    if isnan(val)
        out = '--';
    else
        out = sprintf('%.2f', val);
    end
end

%% Helper function for PID simulation
function [peak_pid, settle_pid_ms, rms_pid] = run_PID_simulation(Kp, Kd, Ki, ...
    W0, V0, A0, f_dyn, Mf, Cf, Kf, Keff, dt, Nt, t, mid_dof, freeDOF, ...
    totalDOF, dof_per_node, left_node, right_node, peak_response)

    nDOF_f = size(W0, 1);
    W_pid = W0; V_pid = V0; A_pid = A0;
    f_pid = f_dyn;
    M_record = zeros(1, Nt);
    curvature_integral = 0;

    thetaL_global = (left_node - 1)*dof_per_node + 3;
    thetaR_global = (right_node - 1)*dof_per_node + 3;
    idL = find(freeDOF == thetaL_global); idR = find(freeDOF == thetaR_global);

    b = 1/4; g = 1/2;
    a0 = 1/(b*dt^2); a1 = g/(b*dt); a2 = 1/(b*dt); a3 = 1/(2*b)-1;
    a4 = g/b - 1; a5 = dt*(g/(2*b)-1);

    control_threshold = 0.075 * peak_response;   
    sustain_steps = round(0.005 / dt);          
    below_thresh_counter = 0;
    control_active = true;
    decay_factor = 1.0;
    decay_tau = 0.005;  
    decay_started = false;

    for n = 1:Nt
        % Check midpoint DOF magnitude
        if abs(W_pid(mid_dof, n)) < control_threshold
            below_thresh_counter = below_thresh_counter + 1;
        else
            below_thresh_counter = 0;
        end
    
        % Trigger decay if sustained below threshold
        if control_active && below_thresh_counter >= sustain_steps
            control_active = false;
            decay_started = true;
            decay_start_time = t(n);
        end
    
        % Update decay factor if in decay mode
        if decay_started
            elapsed_decay = t(n) - decay_start_time;
            decay_factor = exp(-elapsed_decay / decay_tau);
            % Optional: zero control if it’s negligible
            if decay_factor < 1e-4
                decay_factor = 0;
            end
        else
            decay_factor = 1.0;
        end
    
        % Control moment calculation (decay applied to full law)
        thetaL = W_pid(idL, n); thetaR = W_pid(idR, n);
        dthetaL = V_pid(idL, n); dthetaR = V_pid(idR, n);
    
        curvature = thetaR - thetaL;
        curvature_rate = dthetaR - dthetaL;
        curvature_integral = curvature_integral + curvature * dt;
    
        M_control = decay_factor * (-Kp * curvature - Kd * curvature_rate - Ki * curvature_integral);
    
        % Apply moment couple
        f_pid(idL, n+1) = f_pid(idL, n+1) - M_control;
        f_pid(idR, n+1) = f_pid(idR, n+1) + M_control;
    
        M_record_pid(n) = M_control;
    
        % Newmark Integration
        Feff = f_pid(:,n+1) + ...
            Mf*(a0*W_pid(:,n) + a2*V_pid(:,n) + a3*A_pid(:,n)) + ...
            Cf*(a1*W_pid(:,n) + a4*V_pid(:,n) + a5*A_pid(:,n));
    
        W_pid(:,n+1) = Keff \ Feff;
        A_pid(:,n+1) = a0*(W_pid(:,n+1)-W_pid(:,n)) - a2*V_pid(:,n) - a3*A_pid(:,n);
        V_pid(:,n+1) = V_pid(:,n) + dt*((1-g)*A_pid(:,n) + g*A_pid(:,n+1));
    end

    response = W_pid(mid_dof, :);
    peak_pid = max(abs(response));
    rms_pid = rms(A_pid(mid_dof, t <= 0.015));

    % Settling time detection
    settle_thresh = 0.05 * peak_pid;
    sustain_steps = round(0.005 / dt);
    [~, peak_idx] = max(abs(response));
    settle_pid_ms = NaN;

    for k = peak_idx:Nt - sustain_steps
        if all(abs(response(k:k + sustain_steps)) < settle_thresh)
            settle_pid_ms = (k * dt) * 1e3;
            break;
        end
    end
end
