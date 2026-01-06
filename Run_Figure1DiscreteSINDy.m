
% ------------------------------------------------------------
% DATA GEN: Distributed delay via Linear Chain Trick (gamma/Erlang kernel)
%            Choose p_true and tau_mean_true (so a_true = p_true / tau_mean_true).
% IDENTIFICATION: Discrete-delay SINDy:
%   Library: {1, M, P, M^2, M*P, P^2, Hill(P(t - tau); K, n)}
%   - ONLY lagged feature is Hill(P_tau)
%   - No noise, 
%   - Model selection by BIC across grid of tau (K,n fixed here)
%
% This tests if a discrete-delay library can approximate a distributed delay.


clear; close all; clc;

%% Ground-truth (GENERATOR = DISTRIBUTED DELAY) 
Mm_true      = 0.03;      % mRNA degradation
Mp_true      = 0.03;      % protein degradation
alpha_m_true = 1.0;       % max transcription rate
alpha_p_true = 2.0;       % translation rate
K_true       = 100;       % Hill threshold
nH_true      = 5;         % Hill exponent 

% Distributed delay for generation
p_true          = 100;      % chain order 
tau_mean_true   = 20;     % mean delay 
a_true          = p_true / tau_mean_true;  % chain rate

% History / initial conditions 
hist_M = 3;
hist_P = 100.0;
z0     = hist_P * ones(p_true,1);   % start each chain state at history P
y0     = [hist_M; hist_P; z0];      

%% Generate training data 

odeFun = @(t,y) hes1_lct_ode(t,y,Mm_true,Mp_true,alpha_m_true,alpha_p_true,...
                             K_true,nH_true,p_true,a_true);

tspan = [0, 800];
opts  = odeset('RelTol',1e-8,'AbsTol',1e-10);

sol   = ode15s(odeFun, tspan, y0, opts);

% Uniform sampling
dt   = 0.02;
t    = (tspan(1):dt:tspan(2))';
Y    = deval(sol, t);               
M    = Y(1,:)';
P    = Y(2,:)';
Zp   = Y(2 + p_true, :)';           
N    = numel(t);

% Numerical derivatives 
dM = finite_diff(M, dt);
dP = finite_diff(P, dt);

%%  Identification settings (DISCRETE-DELAY SINDy) 
% Tau grid to test discrete-delay approximation of the distributed delay
tau_grid = 10:1:30;     
% Fix Hill K and n (you can grid these later if desired)
K_grid   = 100;
n_grid   = 5;

% Single sparsity threshold 
lambda = 1;           

% Polynomial order for instantaneous library
polyorder = 1;

%% Search over tau by BIC 
best = struct('BIC', inf, 'tau', NaN, 'K', NaN, 'n', NaN, ...
              'lambda', NaN, 'XiM', [], 'XiP', [], 'features', [], ...
              'Theta_cols', [], 'res', [], 'rss', NaN, 'k', NaN);

for tau = tau_grid
    % delayed P for the SINDy library (discrete-delay feature)
    P_tau = delayed_signal(t, P, tau);

    for Kc = K_grid
        for nc = n_grid
            % Hill only on delayed P
            H = 1 ./ (1 + (P_tau./Kc).^nc);

            % Library
            [Theta, labels] = build_library(M, P, H, polyorder);

            % Normalize columns (for threshold stability)
            [ThetaN, col_scales] = normalize_columns(Theta);

            % Fit dM/dt and dP/dt with a single lambda
            XiM = stlsq(ThetaN, dM, lambda, 10);
            XiP = stlsq(ThetaN, dP, lambda, 10);

            % Unscale
            XiM = XiM ./ col_scales(:);
            XiP = XiP ./ col_scales(:);

            % Residuals/BIC
            rM  = dM - Theta*XiM;
            rP  = dP - Theta*XiP;
            res = [rM; rP];
            rss = sum(res.^2);
            k   = nnz(abs(XiM) > 0) + nnz(abs(XiP) > 0);
            N_eff = 2*N;
            BIC = N_eff*log(rss/max(N_eff,1)) + k*log(N_eff);

            if isfinite(BIC) && BIC < best.BIC
                best.BIC      = BIC;
                best.tau      = tau;
                best.K        = Kc;
                best.n        = nc;
                best.lambda   = lambda;
                best.XiM      = XiM;
                best.XiP      = XiP;
                best.features = labels;
                best.Theta_cols = find(abs(XiM)>0 | abs(XiP)>0);
                best.res      = res;
                best.rss      = rss;
                best.k        = k;
            end
        end
    end
end

%% Report results 
fprintf('\n===== SINDy-DDE (Hes1) — Best Discrete-Delay Approx to Distributed Data =====\n');
fprintf('  (Generator) gamma/Erlang: p = %d, tau_mean = %.3f, a = %.5f\n', p_true, tau_mean_true, a_true);
fprintf('  (Identifier) tau_hat = %.3f  (vs mean %.3f)\n', best.tau, tau_mean_true);
fprintf('  K_hat   = %.3f  (true %.3f)\n', best.K,   K_true);
fprintf('  n_hat   = %d     (true %d)\n',   best.n,   nH_true);
fprintf('  lambda  = %.3f\n', best.lambda);
fprintf('  BIC     = %.3f,  rss = %.3e,  k = %d\n', best.BIC, best.rss, best.k);

fprintf('\n--- Discovered terms for dM/dt ---\n');
print_model(best.features, best.XiM);

fprintf('\n--- Discovered terms for dP/dt ---\n');
print_model(best.features, best.XiP);

fprintf('\n(Only the library Hill term uses delayed P; generator used distributed delay via z_p(t).)\n');

%%  Compare trajectories: distributed truth vs identified 
% Simulate the identified discrete-delay model with dde23 and compare.
historyFun = @(t) [hist_M; hist_P];    % for learned discrete-delay DDE
learned_dde = @(t,y,Z) learned_dde_rhs(t,y,Z,best,polyorder);
sol_id = dde23(learned_dde, best.tau, historyFun, tspan, opts);
Yhat   = deval(sol_id, t);
Mhat   = Yhat(1,:)'; 
Phat   = Yhat(2,:)';
% TRAJECTORY residuals 
eM = M - Mhat;
eP = P - Phat;

rss_traj_M = sum(eM.^2);
rss_traj_P = sum(eP.^2);
rss_traj_total = rss_traj_M + rss_traj_P;

% Trajectory metrics 
fprintf('\n= Trajectory metrics (BEST model) =\n');

% RSS (trajectory)
fprintf('RSS  M (traj)        = %.6e\n', rss_traj_M);
fprintf('RSS  P (traj)        = %.6e\n', rss_traj_P);
fprintf('RSS  sum (traj)      = %.6e\n', rss_traj_total);

% Relative L2 trajectory error
denM = norm(M,2); if denM==0, denM=eps; end
denP = norm(P,2); if denP==0, denP=eps; end
relL2_M = norm(eM,2)/denM;
relL2_P = norm(eP,2)/denP;

fprintf('Rel L2 (M)           = %.6e\n', relL2_M);
fprintf('Rel L2 (P)           = %.6e\n', relL2_P);

% BIC per trajectory component (reporting only)
Ntraj = numel(eM);
mse_traj_M = rss_traj_M / Ntraj;
mse_traj_P = rss_traj_P / Ntraj;

thrBIC = 1e-6;   
kM_traj = nnz(abs(best.XiM) > thrBIC);
kP_traj = nnz(abs(best.XiP) > thrBIC);

BIC_traj_M   = Ntraj*log(mse_traj_M + 1e-12) + kM_traj*log(Ntraj);
BIC_traj_P   = Ntraj*log(mse_traj_P + 1e-12) + kP_traj*log(Ntraj);
BIC_traj_sum = BIC_traj_M + BIC_traj_P;

fprintf('BIC  (M) traj        = %.6f\n', BIC_traj_M);
fprintf('BIC  (P) traj        = %.6f\n', BIC_traj_P);
fprintf('BIC  sum traj        = %.6f\n', BIC_traj_sum);
fprintf('Active terms (traj): k_M=%d, k_P=%d\n', kM_traj, kP_traj);

% Plot comparison 
figure('Color','w'); 
tiledlayout(2,1,'Padding','compact','TileSpacing','compact');

nexttile;
plot(t, P, '-', 'LineWidth', 1.2); hold on;
plot(t, Phat, '--', 'LineWidth', 1.2);
xlabel('t'); ylabel('H(t)');
title(sprintf('', ...
      p_true, tau_mean_true, best.tau));
legend('Truth (LCT)','Identified (DDE)','Location','best'); grid on;

nexttile;
plot(t, M, '-', 'LineWidth', 1.2); hold on;
plot(t, Mhat, '--', 'LineWidth', 1.2);
xlabel('t'); ylabel('M(t)');
title(''); 
legend('Truth (LCT)','Identified (DDE)','Location','best'); grid on;

%%  Helper: LCT ODE (distributed delay generator) 
function dydt = hes1_lct_ode(~, y, Mm, Mp, alpha_m, alpha_p, K, nH, p, a)
    % y = [M; P; z1; z2; ...; zp]  with z chain driven by P
    M  = y(1); 
    P  = y(2);
    z  = y(3:end);
   
    dz = zeros(p,1);
    dz(1) = a*(P       - z(1));
    for i = 2:p
        dz(i) = a*(z(i-1) - z(i));
    end
    zp = z(end); 
    H  = 1/(1 + (zp/K)^nH);
    dM = alpha_m*H - Mm*M;
    dP = alpha_p*M - Mp*P;
    dydt = [dM; dP; dz];
end

%% Helper: Finite differences 
function dx = finite_diff(x, dt)
    n = numel(x);
    dx = zeros(n,1);
    if n < 3, dx(:) = 0; return; end
    dx(1)   = (x(2)   - x(1))   / dt;
    dx(end) = (x(end) - x(end-1))/ dt;
    dx(2:end-1) = (x(3:end) - x(1:end-2)) / (2*dt);
end

%% Helper: Delayed signal via interpolation 
function x_tau = delayed_signal(t, x, tau)
    t_del = t - tau;
    x_tau = interp1(t, x, t_del, 'linear', 'extrap');
    early = t_del < t(1);
    if any(early), x_tau(early) = x(1); end
end

%% Helper: Build library 
function [Theta, labels] = build_library(M, P, H, polyorder)
    Phi = cell(0,1); names = {};
    Phi{end+1} = ones(size(M)); names{end+1} = '1';
    Phi{end+1} = M; names{end+1} = 'M';
    Phi{end+1} = P; names{end+1} = 'P';
    if polyorder >= 2
        Phi{end+1} = M.^2;    names{end+1} = 'M^2';
        Phi{end+1} = M.*P;    names{end+1} = 'M*P';
        Phi{end+1} = P.^2;    names{end+1} = 'P^2';
    end
    if polyorder >= 3
        Phi{end+1} = M.^3;    names{end+1} = 'M^3';
        Phi{end+1} = M.^2.*P; names{end+1} = 'M^2*P';
        Phi{end+1} = M.*P.^2; names{end+1} = 'M*P^2';
        Phi{end+1} = P.^3;    names{end+1} = 'P^3';
    end
    % ONLY lagged feature:
    Phi{end+1} = H; names{end+1} = 'Hill(P_tau)';
    Theta  = [Phi{:}];
    labels = names(:);
end

%% Helper: Normalize columns 
function [ThetaN, scales] = normalize_columns(Theta)
    scales = sqrt(sum(Theta.^2,1))';
    scales(scales == 0) = 1;
    ThetaN = Theta ./ scales';
end

%% Helper: STLSQ 
function Xi = stlsq(Theta, dX, lambda, nIter)
    ridge = 1e-6;
    Xi = (Theta.'*Theta + ridge*eye(size(Theta,2))) \ (Theta.'*dX);
    for k = 1:nIter
        small = abs(Xi) < lambda;
        Xi(small) = 0;
        big = ~small;
        if any(big)
            Xi(big) = (Theta(:,big).' * Theta(:,big) + ridge*eye(nnz(big))) \ ...
                       (Theta(:,big).' * dX);
        end
    end
end

%% Helper: RHS of learned model for dde23 simulation 
function dydt = learned_dde_rhs(~, y, Z, best, polyorder)
    M  = y(1);  P  = y(2);
    Pt = Z(2);  
    H  = 1/(1 + (Pt / best.K)^best.n);
    phi = [];
    phi(end+1,1) = 1;      
    phi(end+1,1) = M;     
    phi(end+1,1) = P;      
    if polyorder >= 2
        phi(end+1,1) = M^2;
        phi(end+1,1) = M*P;
        phi(end+1,1) = P^2;
    end
    if polyorder >= 3
        phi(end+1,1) = M^3;
        phi(end+1,1) = M^2*P;
        phi(end+1,1) = M*P^2;
        phi(end+1,1) = P^3;
    end
    phi(end+1,1) = H;      
    dM = phi.' * best.XiM;
    dP = phi.' * best.XiP;
    dydt = [dM; dP];
end

%%  Helper: print model
function print_model(labels, Xi)
    idx = find(abs(Xi) > 0);
    if isempty(idx)
        fprintf('  (no active terms)\n'); return;
    end
    for k = 1:numel(idx)
        j = idx(k);
        fprintf('  %+ .6f * %-12s\n', Xi(j), labels{j});
    end
end
