%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Logistic_example.m
% IMPORTANT: To reproduce the reported results, the Savitzky–Golay
% window size must be set to 2.5.
% Selection :
%   Use validation / cross-validation on TRAJECTORY to select (n_chain, tau).
%   - Fit Xi using TRAIN portion of the time series.
%   - For each candidate (n_chain, tau), roll out the identified model and
%     score it on the VALIDATION portion using trajectory error + complexity.
%   - Plot results with a vertical line showing the TRAIN/VAL split.
%
% Model used to generate synthetic data (distributed delay logistic):
%   x'(t) = r x(t) (1 - z_p(t)/K),  z_p is terminal state of Erlang/LCT chain.
%
% REQUIRED HELPERS in 'utils':
%   rhs_dist_logistic_LCT.m
%   estimate_derivatives_scalar.m
%   build_LCT_chain_from_signal.m
%   build_logistic_library.m
%   STRidge_sweep.m
%   rollout_logistic_LCT_SINDy.m
%   printSINDyEq.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;
addpath(genpath('utils'));  

%% USER SETTINGS 
% Library / regression
polyorder    = 2;         
normalize_dX = true;

% Derivative mode: 'raw' | 'sgolay' 
smooth_deriv = 'sgolay';

% Reference trajectory used only for scoring (train + validation)
%   'noisy'      -> x_ref = x        (measured)
%   'smoothed'   -> x_ref = x_s      (denoised)
%   'groundtruth'-> x_ref = x_true   (clean synthetic)
scoring_mode = 'smoothed';

% STRidge parameters
lambda_thresh = 0.5;
ridge_list    = logspace(-6, 0, 7);
max_iter_str  = 10;

% Sampling / integration
dt   = 0.1;
Tend = 100;

% TRAIN / VALIDATION split (trajectory CV)
train_frac = 0.4;                  
t_split    = train_frac * Tend;     

% Ground truth parameters (for DATA GENERATION ONLY)
r_true   = 0.5;
K_true   = 50;

p_true   = 10;
tau_true = 4;
a_true   = p_true / tau_true;

x0       = 10;
z0_true  = repmat(x0, p_true, 1);

% Search grids (IDENTIFICATION)
n_list   = 1:1:10;
tau_grid = 1:1:10;

% Complexity penalties 
bic_penalty_per_chain = 0.01;   % extra penalty for larger chain

% Solver tolerances
optsZ  = odeset('RelTol',1e-9,'AbsTol',1e-11);
optsGT = odeset('RelTol',1e-9,'AbsTol',1e-11);
optsO  = odeset('RelTol',1e-4,'AbsTol',1e-6);

%% --------- 1) Generate ground-truth data (distributed delay) ----------
rhsGT = @(t,y) rhs_dist_logistic_LCT(y, r_true, K_true, a_true, p_true);

y0 = [x0; z0_true];
[tGT, yGT] = ode45(rhsGT, [0 Tend], y0, optsGT);

% Uniform sampling on [0, Tend] with step dt
t      = (0:dt:Tend).';
yS     = interp1(tGT, yGT, t, 'linear');
x_true = yS(:,1);

% Measured data (add measurement noise)
noise_level = 0.5;
rng(0)
x = x_true + noise_level * std(x_true) * randn(size(x_true));

%% 2) Estimate derivatives dx/dt 
[dx, x_s] = estimate_derivatives_scalar(t, x, smooth_deriv);

x_all_obs    = x;
x_all_smooth = x_s;

% Choose reference trajectory for scoring
switch lower(scoring_mode)
    case 'smoothed'
        x_ref_all = x_all_smooth;
    case 'groundtruth'
        x_ref_all = x_true;
    otherwise
        x_ref_all = x_all_obs;
end

% Trim edges to avoid derivative boundary artifacts
idx_all   = 2:numel(t)-1;
t_use     = t(idx_all);
x_use     = x_all_smooth(idx_all);   % use smoothed states to build library
dx_use    = dx(idx_all);
xref_use  = x_ref_all(idx_all);

% Define TRAIN/VAL masks on the trimmed grid
isTrain = (t_use <= t_split);
isVal   = (t_use >  t_split);

if nnz(isTrain) < 10 || nnz(isVal) < 10
    error('Train/Val split too extreme. Adjust train_frac or dt/Tend.');
end

t_train  = t_use(isTrain);
x_train0 = x_use(isTrain);
dx_train = dx_use(isTrain);

t_val    = t_use(isVal);
xref_val = xref_use(isVal);

%% 3) Grid-search over (n_chain, tau) with VAL trajectory score 
best  = struct([]);
tried = 0;
kept  = 0;

cand = struct('n',{},'tau',{},'a_chain',{},'Xi',{}, ...
              'BIC_val',{},'mse_val',{},'rms_val',{}, ...
              'BIC_train',{},'mse_train',{},'k',{},'termNames',{});

thrK = 1e-6;   % active term threshold 

for n_chain = n_list
    for tau = tau_grid
        a_chain = n_chain / tau;

        % Build surrogate LCT chain from full smoothed signal x_s
        % (then slice train/val consistently)
        zn_full_trim = build_LCT_chain_from_signal(t, x_s, n_chain, tau, optsZ);
        zn_use       = zn_full_trim(idx_all);

        zn_train = zn_use(isTrain);

        % Build library on TRAIN only
        [Theta_train, termNames] = build_logistic_library(x_train0, zn_train, polyorder);

        % Normalize features 
        colscale              = vecnorm(Theta_train, 2, 1);
        colscale(colscale==0) = 1;
        ThetaN                = Theta_train ./ colscale;

        % Normalize target derivative 
        if normalize_dX
            dx_scale = norm(dx_train, 2);
            if dx_scale == 0, dx_scale = 1; end
            dxN = dx_train ./ dx_scale;
        else
            dx_scale = 1;
            dxN      = dx_train;
        end

        % SINDy on TRAIN
        tried = tried + 1;
        [XiN, ~, ~] = STRidge_sweep(ThetaN, dxN, ridge_list, lambda_thresh, max_iter_str);

        % Un-normalize Xi back to original scale
        Xi = (XiN ./ colscale.') * dx_scale;

        % Roll out identified model over full horizon 
        par_roll = struct('polyorder',polyorder, ...
                          'n_chain',n_chain, ...
                          'a_chain',a_chain, ...
                          'Xi',Xi);

        y0_roll = [x0; repmat(x_s(1), n_chain, 1)];

        try
            [tBraw, yBraw] = ode15s(@(tt,yy) rollout_logistic_LCT_SINDy(tt,yy,par_roll), ...
                                   [0 t(end)], y0_roll, optsO);
        catch
            continue;
        end

        x_id_full = interp1(tBraw, yBraw(:,1), t, 'linear');
        x_id_use  = x_id_full(idx_all);

        % TRAIN trajectory error 
        e_train = x_id_use(isTrain) - xref_use(isTrain);
        goodT   = isfinite(e_train);
        e_train = e_train(goodT);
        if isempty(e_train), continue; end
        mse_train = mean(e_train.^2);

        % VAL trajectory error (this drives selection) 
        e_val = x_id_use(isVal) - xref_val;
        goodV = isfinite(e_val);
        e_val = e_val(goodV);
        if isempty(e_val), continue; end

        kept    = kept + 1;
        mse_val = mean(e_val.^2);
        rms_val = sqrt(mse_val);

        % Complexity (active terms)
        k = nnz(abs(Xi) > thrK);

        % BIC-style score on VALIDATION trajectory
        Nval    = numel(e_val);
        BIC_val = Nval*log(mse_val + 1e-12) + k*log(Nval) + bic_penalty_per_chain*n_chain;

        %  BIC-style score on TRAIN trajectory for diagnostics
        Ntr       = numel(e_train);
        BIC_train = Ntr*log(mse_train + 1e-12) + k*log(Ntr) + bic_penalty_per_chain*n_chain;

        cand(end+1) = struct( ... 
            'n',n_chain,'tau',tau,'a_chain',a_chain, ...
            'Xi',Xi,'BIC_val',BIC_val,'mse_val',mse_val,'rms_val',rms_val, ...
            'BIC_train',BIC_train,'mse_train',mse_train,'k',k, ...
            'termNames',{termNames});

        if isempty(best) || BIC_val < best.BIC_val
            best = cand(end);
        end
    end
end

fprintf('Tried %d combos; kept %d.\n', tried, kept);
if isempty(best)
    error('No valid combos. Adjust grids or tolerances.');
end

%% 4) Show top models (by VALIDATION score) 
[~,ord] = sort([cand.BIC_val]);
cand    = cand(ord);

Kshow = min(5, numel(cand));
fprintf('\n=== Top %d candidates (by VAL-BIC) ===\n', Kshow);
for i = 1:Kshow
    c = cand(i);
    fprintf('%2d) n=%-3d  tau=%6.2f  a=%7.3f | ', i, c.n, c.tau, c.a_chain);
    fprintf('RMS_val=%.3g | BIC_val=%.2f | active=%d\n', c.rms_val, c.BIC_val, c.k);
end

Xi        = best.Xi;
termNames = best.termNames;

thr_show = 1e-6;
Xi_print = Xi;
Xi_print(abs(Xi_print) < thr_show) = 0;

fprintf('\n=== BEST model (selected by VALIDATION trajectory) ===\n');
fprintf('n = %d, tau = %.6g, a = n/tau = %.6g | ', best.n, best.tau, best.a_chain);
fprintf('RMS_val = %.3g | BIC_val = %.2f | active = %d\n', best.rms_val, best.BIC_val, best.k);
fprintf('BIC_train = %.2f | MSE_train = %.3e | MSE_val = %.3e\n', best.BIC_train, best.mse_train, best.mse_val);

try
    T = array2table(Xi_print, 'RowNames', termNames, 'VariableNames', {'dxdt'});
    disp('--- SINDy Coefficient Vector Xi ---');
    disp(T);
catch
    disp('--- SINDy Coefficient Vector Xi ---');
    disp(Xi_print);
end

fprintf('\nIdentified BEST model:\n');
fprintf('dx/dt = '); printSINDyEq(Xi(:,1), termNames); fprintf('\n');

%% 5) Final rollout with best model 
par_best = struct('polyorder',polyorder, ...
                  'n_chain',best.n, ...
                  'a_chain',best.a_chain, ...
                  'Xi',best.Xi);

y0_best = [x0; repmat(x_s(1), best.n, 1)];
[tBbest, yBbest] = ode15s(@(tt,yy) rollout_logistic_LCT_SINDy(tt,yy,par_best), ...
                         [0 t(end)], y0_best, optsO);

x_id_best = interp1(tBbest, yBbest(:,1), t, 'linear');

%% 6) Error metrics (report TRAIN and VAL trajectory) 
% Use the same reference as scoring_mode
x_ref_final = x_ref_all;

% Trim to match masks
xid_use   = x_id_best(idx_all);
xref_use2 = x_ref_final(idx_all);

e_train = xid_use(isTrain) - xref_use2(isTrain);
e_val   = xid_use(isVal)   - xref_use2(isVal);

e_train = e_train(isfinite(e_train));
e_val   = e_val(isfinite(e_val));

rss_train = sum(e_train.^2);
rss_val   = sum(e_val.^2);

relL2_train = norm(e_train,2) / max(norm(xref_use2(isTrain),2), 1e-12);
relL2_val   = norm(e_val,2)   / max(norm(xref_use2(isVal),2),   1e-12);

fprintf('\n--- TRAJECTORY METRICS ---\n');
fprintf('TRAIN: RSS = %.3e | RelL2 = %.3e | RMS = %.3e\n', rss_train, relL2_train, sqrt(mean(e_train.^2)));
fprintf('VAL  : RSS = %.3e | RelL2 = %.3e | RMS = %.3e\n', rss_val,   relL2_val,   sqrt(mean(e_val.^2)));

%% 7) FIGURE (a): Training data (noisy vs smoothed) 
figure('Color','w');
plot(t, x, '.', 'MarkerSize', 6); hold on;
plot(t, x_s, '-', 'LineWidth', 1.2);
xline(t_split, '--', 'LineWidth', 1.2);  
grid on;
xlabel('Time');
ylabel('x(t)');
legend('Measured x (noisy)', 'Smoothed x_s', 'Train/Val split', 'Location','best');

%% 8) FIGURE (b): Distributed truth vs identified 
figure('Color','w');
plot(t, x_true, '-',  'LineWidth', 1.0); hold on;
plot(t, x_id_best,'--','LineWidth', 1.4);
xline(t_split, '--', 'LineWidth', 1.2);  % <-- TRAIN/VAL split
grid on;
xlabel('Time');
ylabel('x(t)');
legend('Truth (LCT)', 'Identified model', 'Train/Val split', 'Location','best');


% END Figure_logistic_LCT_SINDy_experiment_CV.m

