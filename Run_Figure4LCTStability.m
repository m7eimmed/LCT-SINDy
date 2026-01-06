%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure4LCTStability.m
%
% Goal:
%   Test how sampling interval (dt) and measurement noise (eta) affect
%   the reconstruction of:
%       - the last LCT chain state z_p (delayed state),
%       - the SG-estimated derivatives dM/dt and dP/dt.
%
%   We:
%     1) Generate dense ground-truth Hes1 data from a distributed-delay
%        model using an LCT chain (Erlang kernel).
%     2) For a range of dt, resample, compute relative errors vs ground truth.
%     3) For a range of noise levels, with dt fixed, do the same.
%
% Requirements:
%   - rhs_dist_Hes1_LCT.m
%   - build_LCT_chain_from_P.m
%   - estimate_derivatives.m   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;
addpath(genpath('utils'));

%% 1) Model parameters 

% Ground-truth parameters (distributed delay model, for data generation)
Mm       = 0.03;
Mp       = 0.03;
P_0_true = 100;
N_true   = 5;
alpha_m  = 1;
alpha_p  = 2;

p_true   = 10;          % true Erlang order (chain length)
tau_true = 20;          % true mean delay
a_true   = p_true / tau_true;

M0 = 3;
P0 = 100;
z0_true = repmat(P0, p_true, 1);   % initialize chain with P0

% Simulation horizon and ODE options
Tend  = 800;
optsGT = odeset('RelTol',1e-9,'AbsTol',1e-11);

% Right-hand side for dense ground-truth simulation
rhsGT = @(t,y) rhs_dist_Hes1_LCT(y, Mm, Mp, P_0_true, N_true, ...
                                 alpha_m, alpha_p, a_true, p_true);

% Dense time grid for ground truth
tGT = linspace(0, Tend, 8001).';   % 0.1-step nominal (adjust if desired)
y0  = [M0; P0; z0_true];

[tGT, yGT] = ode45(rhsGT, tGT, y0, optsGT);

% Ground-truth states
M_GT = yGT(:,1);
P_GT = yGT(:,2);

% Ground-truth last-chain state z_p 
z_p_true_dense = yGT(:, 2 + p_true);   % state index: 1=M,2=P, then chain

%%  relative RMS error
eps_den = 1e-12;  % small epsilon for robustness
relerr = @(yhat,ytrue) ...
    sqrt(mean((yhat - ytrue).^2)) / (sqrt(mean(ytrue.^2)) + eps_den);


%% 2) Error vs sampling (dt) 
% Noise fixed 0 to isolate sampling effect.

dt_list     = [1 2 3 5 8 12 15 20];   % sampling intervals (minutes)
noise_fixed = 0;                       % relative noise level
rng(2025);                             % reproducible seed

% Preallocate arrays for relative errors
Rz_vs_dt  = zeros(size(dt_list));   % z_p relative error
RdM_vs_dt = zeros(size(dt_list));   % dM/dt relative error (from SG)
RdP_vs_dt = zeros(size(dt_list));   % dP/dt relative error (from SG)

% ODE options for LCT chain reconstruction
optsZ = odeset('RelTol',1e-9,'AbsTol',1e-11);

for i = 1:numel(dt_list)
    dt_i = dt_list(i);

    % Sample ground truth onto this coarser grid
    t_i  = (0:dt_i:Tend).';
    yS_i = interp1(tGT, yGT, t_i, 'linear');
    M_i  = yS_i(:,1);
    P_i  = yS_i(:,2);

    % True derivatives [dM; dP] on this grid from RHS 
    dXi_true = zeros(numel(t_i),2);
    for k = 1:numel(t_i)
        fi = rhsGT(t_i(k), yS_i(k,:).');
        dXi_true(k,:) = fi(1:2).';   
    end

    % Add measurement noise to M and P 
    M_meas_i = M_i + noise_fixed*std(M_i)*randn(size(M_i));
    P_meas_i = P_i + noise_fixed*std(P_i)*randn(size(P_i));

    % Savitzky–Golay: denoise P (for LCT)
    [dMdt_sg, dPdt_sg, ~, P_s_forLCT] = ...
        estimate_derivatives(t_i, M_meas_i, P_meas_i, 'sgolay');

    % Reconstruct z_p with TRUE (p_true, tau_true) to isolate LCT effect
    z_chain_i = build_LCT_chain_from_P(t_i, P_s_forLCT, p_true, tau_true, optsZ);
    z_id_i    = z_chain_i(:,end);

    % True z_p on this grid
    z_true_i = interp1(tGT, z_p_true_dense, t_i, 'linear');

    % Aggregate relative errors (RMS-normalized)
    Rz_vs_dt(i)  = relerr(z_id_i,   z_true_i);
    RdM_vs_dt(i) = relerr(dMdt_sg,  dXi_true(:,1));
    RdP_vs_dt(i) = relerr(dPdt_sg,  dXi_true(:,2));
end

% Plot: robustness vs sampling (dt)
figure('Name','Robustness vs sampling (dt)');
plot(dt_list, Rz_vs_dt , '-o','LineWidth',1.6,'MarkerSize',6); hold on;
plot(dt_list, RdM_vs_dt, '-s','LineWidth',1.6,'MarkerSize',6);
plot(dt_list, RdP_vs_dt, '-^','LineWidth',1.6,'MarkerSize',6);
grid on; xlabel('\Delta t (min)'); ylabel('Relative error (RMS-normalized)');
title(sprintf('Reconstruction stability vs sampling (noise = %.2f)', noise_fixed));
legend('z_p (LCT output)', 'dM/dt (SG)', 'dP/dt (SG)', 'Location','best');
hold off;


%% 3) Error vs noise 
% Sampling fixed; vary relative noise level.

dt_fixed   = 5;   % choose a representative sampling interval 
noise_list = [0 0.02 0.05 0.1 0.2 0.5 1.0];
rng(314159);      % reproducible seed

% Sample ground truth to this fixed grid once
t_fix = (0:dt_fixed:Tend).';
yS_fx = interp1(tGT, yGT, t_fix, 'linear');
M_fx  = yS_fx(:,1);
P_fx  = yS_fx(:,2);

% True derivatives and true z_p on this fixed grid
dX_true_fx = zeros(numel(t_fix),2);
for k = 1:numel(t_fix)
    fi = rhsGT(t_fix(k), yS_fx(k,:).');
    dX_true_fx(k,:) = fi(1:2).';     % [dM, dP]
end
z_true_fx = interp1(tGT, z_p_true_dense, t_fix, 'linear');

% Preallocate
Rz_vs_noise  = zeros(size(noise_list));
RdM_vs_noise = zeros(size(noise_list));
RdP_vs_noise = zeros(size(noise_list));

for j = 1:numel(noise_list)
    eta = noise_list(j);

    % Add noise to states
    M_meas_fx = M_fx + eta*std(M_fx)*randn(size(M_fx));
    P_meas_fx = P_fx + eta*std(P_fx)*randn(size(P_fx));

    % SG: denoise P for LCT + derivatives
    [dMdt_sg, dPdt_sg, ~, P_s_forLCT] = ...
        estimate_derivatives(t_fix, M_meas_fx, P_meas_fx, 'sgolay');

    % LCT with TRUE (p_true, tau_true)
    z_chain_fx = build_LCT_chain_from_P(t_fix, P_s_forLCT, p_true, tau_true, optsZ);
    z_id_fx    = z_chain_fx(:,end);

    % Aggregate relative errors
    Rz_vs_noise(j)  = relerr(z_id_fx,  z_true_fx);
    RdM_vs_noise(j) = relerr(dMdt_sg,  dX_true_fx(:,1));
    RdP_vs_noise(j) = relerr(dPdt_sg,  dX_true_fx(:,2));
end

% Plot: robustness vs noise 
figure('Name','Robustness vs noise');
plot(noise_list, Rz_vs_noise , '-o','LineWidth',1.6,'MarkerSize',6); hold on;
plot(noise_list, RdM_vs_noise, '-s','LineWidth',1.6,'MarkerSize',6);
plot(noise_list, RdP_vs_noise, '-^','LineWidth',1.6,'MarkerSize',6);
grid on;
xlabel('Noise level \eta (relative to signal std)');
ylabel('Relative error (RMS-normalized)');
title(sprintf('Reconstruction stability vs noise (\\Delta t = %g min)', dt_fixed));
legend('z_p (LCT output)', 'dM/dt (SG)', 'dP/dt (SG)', 'Location','best');
hold off;


% END Figure4LCTStability.m

