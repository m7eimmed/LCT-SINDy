%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ikeda_Example.m
% IMPORTANT: To reproduce the reported results, the Savitzky–Golay
% window size must be set to 0.5.

% Ikeda (discrete-delay) ground truth (dde23):
%   x'(t) = -x(t) + alpha * sin(x(t - tau_true))
%
% Identification model (LCT approximation with chain length p):
%   z1' = a (x - z1),  zj' = a (z_{j-1} - zj),  a = p/tau
%   x'  = Theta(x, z_p) * Xi   where Theta includes polynomial terms + sin(z_p)
%
% Selection :
%   - Fit Xi on TRAIN derivatives only.
%   - Select (tau) using VALIDATION derivative MSE:
%         dx_val  vs  dxhat_val = Theta_val * Xi
%   - plot truth vs identified trajectory at the end.
%
% Required helpers (in utils/):
%   - rhs_ikeda_dde.m
%   - estimate_derivatives_scalar.m
%   - build_LCT_chain_from_signal.m
%   - build_ikeda_library.m
%   - STRidge_sweep.m  (and STRidge_core.m)
%   - rollout_ikeda_LCT_SINDy.m
%   - printSINDyEq.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;
addpath(genpath('utils'));

%% USER SETTINGS 
polyorder          = 4;          % polynomial library degree in (x,z)
include_cross_sin  = true;       % include x*sin(z), z*sin(z) as extra features
normalize_dX       = true;       % normalize target derivative during STRidge

smooth_deriv  = 'raw';           % 'raw' | 'sgolay' 
scoring_mode  = 'smoothed';      % 'noisy' | 'smoothed' | 'groundtruth' (PLOTS ONLY)

% STRidge parameters
lambda_thresh = 0.5;
ridge_list    = logspace(-6, 0, 7);
max_iter_str  = 10;

% Sampling
dt   = 0.05;
Tend = 50;

% Train/Val split
train_frac = 0.5;
t_split    = train_frac * Tend;

% Ground truth parameters (data generation)
alpha_true = 6;
tau_true   = 1.59;

% DDE history and solver
x0 = 0.1;  % constant history
dde_opts = ddeset('RelTol',1e-9,'AbsTol',1e-11);

% Noise level
noise_level = 0;        
rng(0);

% Identification grids
p_list   = 100;         % we used fixed value for chain length
tau_grid = 1:0.01:2;

% Complexity reporting threshold for active terms
thrK = 1e-6;

% ODE solver tolerances for chain build / rollout
optsZ = odeset('RelTol',1e-9,'AbsTol',1e-11);
optsO = odeset('RelTol',1e-4,'AbsTol',1e-6);

%% 1) Generate ground-truth data using DDE (discrete delay) 
lags   = tau_true;
rhsDDE = @(t, x, xlag) rhs_ikeda_dde(t, x, xlag, alpha_true);
history = @(t) x0;

sol = dde23(rhsDDE, lags, history, [0 Tend], dde_opts);

t = (0:dt:Tend).';
x_true = deval(sol, t).';
x_true = x_true(:);

% Add measurement noise
x = x_true + noise_level * std(x_true) * randn(size(x_true));

%% 2) Estimate derivatives dx/dt 
[dx, x_s] = estimate_derivatives_scalar(t, x, smooth_deriv); % the Savitzky–Golay window size should be set to 0.5.

% Choose reference trajectory for plots only
switch lower(scoring_mode)
    case 'smoothed'
        x_ref_all = x_s;
    case 'groundtruth'
        x_ref_all = x_true;
    otherwise
        x_ref_all = x;
end

% Trim edges to avoid derivative boundary artifacts
idx_all  = 2:numel(t)-1;
t_use    = t(idx_all);

% Use smoothed signal for chain/library
x_use    = x_s(idx_all);

% Use estimated derivative for scoring
dx_use   = dx(idx_all);

% For plotting reference only
xref_use = x_ref_all(idx_all);

% Train/Val masks
isTrain = (t_use <= t_split);
isVal   = (t_use >  t_split);

if nnz(isTrain) < 10 || nnz(isVal) < 10
    error('Train/Val split too extreme. Adjust train_frac or dt/Tend.');
end

t_train  = t_use(isTrain);
x_train  = x_use(isTrain);
dx_train = dx_use(isTrain);

t_val    = t_use(isVal);
x_val    = x_use(isVal);
dx_val   = dx_use(isVal);

%% 3) Grid-search (p, tau) selecting by VALIDATION derivative MSE (NO BIC)
cand = struct('p',{},'tau',{},'a_chain',{},'Xi',{}, ...
              'mse_train',{},'mse_val',{},'rms_val',{}, ...
              'k',{},'termNames',{},'colscale',{},'dx_scale',{});

best  = struct([]);
tried = 0; kept = 0;

for p = p_list
    for tau = tau_grid
        tried = tried + 1;
        a_chain = p / tau;

        % Build LCT terminal state z_p(t) from FULL smoothed signal x_s
        try
            zp_full = build_LCT_chain_from_signal(t, x_s, p, tau, optsZ);
        catch
            continue;
        end
        zp_use = zp_full(idx_all);

        zp_train = zp_use(isTrain);
        zp_val   = zp_use(isVal);

        % Build libraries on train and VAL 
        [Theta_train, termNames] = build_ikeda_library(x_train, zp_train, polyorder, include_cross_sin);
        [Theta_val,   ~]         = build_ikeda_library(x_val,   zp_val,   polyorder, include_cross_sin);

        % Normalize Theta using train only
        colscale              = vecnorm(Theta_train, 2, 1);
        colscale(colscale==0) = 1;

        ThetaN_train = Theta_train ./ colscale;
        ThetaN_val   = Theta_val   ./ colscale;

        % Normalize target derivative (train only)
        if normalize_dX
            dx_scale = norm(dx_train, 2);
            if dx_scale == 0, dx_scale = 1; end
            dxN_train = dx_train ./ dx_scale;
        else
            dx_scale  = 1;
            dxN_train = dx_train;
        end

        % Fit SINDy on TRAIN derivatives
        [XiN, ~, ~] = STRidge_sweep(ThetaN_train, dxN_train, ridge_list, lambda_thresh, max_iter_str);

        % Un-normalize Xi back to original units
        Xi = (XiN ./ colscale.') * dx_scale;

        % Predicted derivatives (TRAIN / VAL) from identified model
        dxhat_train = Theta_train * Xi;
        dxhat_val   = Theta_val   * Xi;

        % TRAIN derivative error 
        e_train = dxhat_train - dx_train;
        e_train = e_train(isfinite(e_train));
        if isempty(e_train), continue; end
        mse_train = mean(e_train.^2);

        % VAL derivative error (SELECTION CRITERION)
        e_val = dxhat_val - dx_val;
        e_val = e_val(isfinite(e_val));
        if isempty(e_val), continue; end
        mse_val = mean(e_val.^2);
        rms_val = sqrt(mse_val);

        % Complexity for reporting only
        k = nnz(abs(Xi) > thrK);

        kept = kept + 1;
        cand(end+1) = struct( ... 
            'p',p,'tau',tau,'a_chain',a_chain,'Xi',Xi, ...
            'mse_train',mse_train,'mse_val',mse_val,'rms_val',rms_val, ...
            'k',k,'termNames',{termNames}, ...
            'colscale',colscale,'dx_scale',dx_scale);

        if isempty(best) || mse_val < best.mse_val
            best = cand(end);
        end
    end
end

fprintf('Tried %d combos; kept %d.\n', tried, kept);
if isempty(best), error('No valid combos. Adjust grids/tolerances.'); end

%% 4) Show top models (by VAL-derivative MSE)
[~,ord] = sort([cand.mse_val]);
cand = cand(ord);

Kshow = min(5, numel(cand));
fprintf('\n=== Top %d candidates (by VAL derivative MSE) ===\n', Kshow);
for i = 1:Kshow
    c = cand(i);
    fprintf('%2d) p=%-3d  tau=%6.3f  a=%7.3f | RMS_val(deriv)=%.3g | MSE_val=%.3g | active=%d\n', ...
        i, c.p, c.tau, c.a_chain, c.rms_val, c.mse_val, c.k);
end

Xi        = best.Xi;
termNames = best.termNames;

Xi_print = Xi;
Xi_print(abs(Xi_print) < 1e-6) = 0;

fprintf('\n=== BEST model (selected by VAL derivative MSE) ===\n');
fprintf('p = %d, tau = %.6g, a = p/tau = %.6g | ', best.p, best.tau, best.a_chain);
fprintf('RMS_val(deriv) = %.3g | MSE_val = %.3g | active = %d\n', best.rms_val, best.mse_val, best.k);

disp('--- SINDy Coefficient Vector Xi ---');
try
    Ttab = array2table(Xi_print, 'RowNames', termNames, 'VariableNames', {'dxdt'});
    disp(Ttab);
catch
    disp(Xi_print);
end

fprintf('\nIdentified BEST model:\n');
fprintf('dx/dt = '); printSINDyEq(Xi(:,1), termNames); fprintf('\n');

%% 5) Final rollout with BEST model 
par_best = struct('polyorder',polyorder, ...
                  'include_cross_sin',include_cross_sin, ...
                  'p_chain',best.p, ...
                  'a_chain',best.a_chain, ...
                  'Xi',best.Xi);

y0_best = [x_s(1); repmat(x_s(1), best.p, 1)];

[tBbest, yBbest] = ode15s(@(tt,yy) rollout_ikeda_LCT_SINDy(tt,yy,par_best), ...
                         [0 t(end)], y0_best, optsO);

x_id_best = interp1(tBbest, yBbest(:,1), t, 'linear');

%% 6) Plots

% (A) Training data only (measured/noisy)
figure('Color','w');
plot(t_train, x(t_use<=t_split), '.', 'MarkerSize', 7);
grid on; xlabel('Time'); ylabel('x(t)');
title(sprintf('Ikeda measured data (TRAIN only): dt=%.3f, noise=%.2f', dt, noise_level));
legend('Measured x (train)', 'Location','best');

% (B) Noisy vs smoothed over full horizon
figure('Color','w');
plot(t, x, '.', 'MarkerSize', 6); hold on;
plot(t, x_s, '-', 'LineWidth', 1.2);
xline(t_split, '--', 'LineWidth', 1.2);
grid on; xlabel('Time'); ylabel('x(t)');
legend('Measured x', 'Smoothed x_s', 'Train/Val split', 'Location','best');

% (C) Truth vs identified trajectory 
figure('Color','w');
plot(t, x_true, '-', 'LineWidth', 1.0); hold on;
plot(t, x_id_best,'--','LineWidth', 1.4);
xline(t_split, '--', 'LineWidth', 1.2);
grid on; xlabel('Time'); ylabel('x(t)');
legend('Truth (DDE)', 'Identified (LCT-SINDy)', 'Train/Val split', 'Location','best');


% END Ikeda_Example.m

