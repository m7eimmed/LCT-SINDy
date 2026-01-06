%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Hes1_mRNA_example.m  
%
% Pipeline:
%   1) Generate synthetic Hes1-mRNA data from distributed-delay model.
%   2) Add measurement noise; denoise + estimate derivatives.
%   3) Define TRAIN/VAL split in time.
%   4) For each candidate (n_chain, tau, K, nH):
%        - Build LCT chain from FULL smoothed P_s (then slice train/val)
%        - Fit Xi on TRAIN derivatives (SINDy/STRidge)
%        - Roll out identified model over FULL horizon
%        - Score on VALIDATION trajectory (BIC-style)
%   5) Report top models by VAL-BIC; final rollout; figures with split line.
%
% Dependencies (in 'utils' on MATLAB path):
%   - rhs_dist_Hes1_LCT.m
%   - estimate_derivatives.m
%   - build_LCT_chain_from_P.m
%   - build_poly_library.m
%   - STRidge_sweep.m
%   - rollout_identified_model.m
%   - printSINDyEq.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear; close all; clc;
addpath(genpath('utils'));  
set(groot, ...
    'DefaultAxesFontSize', 12, ...
    'DefaultAxesLabelFontSizeMultiplier', 1.2, ...
    'DefaultLineLineWidth', 1.5, ...
    'DefaultTextFontSize', 12, ...
    'DefaultLegendFontSize', 12);

%%  USER SETTINGS
% Library / regression
polyorder    = 1;
normalize_dX = true;

% Derivative mode: false | 'sgolay' 
smooth_deriv = 'sgolay';

% Reference trajectory used only for scoring 
%   'noisy' = x_ref = [M  P]      
%   'smoothed' = x_ref = [M_s P_s]  
%   'groundtruth' = x_ref = true solution 
scoring_mode = 'smoothed';

% STRidge
lambda_thresh = 0.5;
ridge_list    = logspace(-6, 0, 7);
max_iter_str  = 10;

% Sampling 
dt   = 0.5;
Tend = 800;

%  TRAIN / VALIDATION split 
train_frac = 0.2;               
t_split    = train_frac*Tend;   

% Ground truth parameters (for DATA GENERATION ONLY)
Mm       = 0.03;
Mp       = 0.03;
P_0_true = 100;
N_true   = 5;
alpha_m  = 1;
alpha_p  = 2;

p_true   = 20;          % true Erlang order
tau_true = 20;          % true mean delay
a_true   = p_true / tau_true;

M0       = 3;
P0       = 100;
z0_true  = repmat(P0, p_true, 1);

% Search grids 
n_list   = 10:1:20;        % candidate chain lengths 
tau_grid = 15:1:25;    % candidate mean delays
n_grid   = 1:1:5;             % Hill exponent
K_grid   = 90:1:100;           % repression threshold

% Complexity penalties
bic_penalty_per_chain = 0.01;  % extra penalty for larger chain
bic_penalty_Kn        = 0;     

% Solver tolerances
optsZ  = odeset('RelTol',1e-9,'AbsTol',1e-11);
optsO  = odeset('RelTol',1e-7,'AbsTol',1e-9);
optsGT = odeset('RelTol',1e-9,'AbsTol',1e-11);

%%  1) Generate ground-truth data 
rhsGT = @(t,y) rhs_dist_Hes1_LCT(y, Mm, Mp, P_0_true, N_true, ...
                                 alpha_m, alpha_p, a_true, p_true);

y0 = [M0; P0; z0_true];
[tGT, yGT] = ode45(rhsGT, [0 Tend], y0, optsGT);

% Uniform sampling on [0, Tend] with step dt
t   = (0:dt:Tend).';
yS  = interp1(tGT, yGT, t, 'linear');   
M_true = yS(:,1);
P_true = yS(:,2);
x_true = [M_true P_true];

% Measured data (add measurement noise)
noise_level = 0.5;
rng(0);                   % reproducible random seed
M = M_true + noise_level * std(M_true) * randn(size(M_true));
P = P_true + noise_level * std(P_true) * randn(size(P_true));

%%  2) Estimate derivatives dM/dt, dP/dt 
[dMdt, dPdt, M_s, P_s] = estimate_derivatives(t, M, P, smooth_deriv);
dx = [dMdt, dPdt];
x_all_obs    = [M  P];
x_all_smooth = [M_s P_s];

% Choose reference trajectory for scoring
switch lower(scoring_mode)
    case 'smoothed'
        x_ref_all = x_all_smooth;
    case 'groundtruth'
        x_ref_all = x_true;
    otherwise
        x_ref_all = x_all_obs;
end

% Trim edges to avoid derivative boundary artifacts.
idx_all  = 2:numel(t)-1;
t_use    = t(idx_all);
x_use    = x_all_smooth(idx_all,:);   % build library from smoothed states 
dx_use   = dx(idx_all,:);
xref_use = x_ref_all(idx_all,:);

% TRAIN/VAL masks on trimmed grid
isTrain = (t_use <= t_split);
isVal   = (t_use >  t_split);

if nnz(isTrain) < 20 || nnz(isVal) < 20
    error('Train/Val split too extreme. Adjust train_frac or dt/Tend.');
end

% TRAIN slices (for fitting Xi)
t_train   = t_use(isTrain); 
x_train0  = x_use(isTrain,:);
dx_train0 = dx_use(isTrain,:);

% VAL reference (for scoring)
xref_val  = xref_use(isVal,:);

%% 3) Grid-search over (n_chain, tau, K, nH) 
best  = struct([]);
tried = 0;
kept  = 0;

thrK = 1e-6;   % active term threshold

cand = struct('n',{},'tau',{},'K',{},'nH',{},'r',{},'Xi',{}, ...
              'BIC_val',{},'mse_val',{},'rmsM_val',{},'rmsP_val',{}, ...
              'BIC_train',{},'mse_train',{},'k',{},'termNames',{});

for n_chain = n_list
    for tau = tau_grid
        r_chain = n_chain / tau;

        % Build surrogate LCT chain from FULL smoothed P_s 
        zn_full_trim = build_LCT_chain_from_P(t, P_s, n_chain, tau, optsZ);
        zn_use       = zn_full_trim(idx_all);

        % Slice chain states for TRAIN
        zn_train = zn_use(isTrain);

        % Base polynomial library Theta(M,P) built on TRAIN only
        [Theta_base, baseNames] = build_poly_library(x_train0, polyorder);

        % Loop over Hill parameters
        for Kc = K_grid
            for nc = n_grid

                % Hill-type repression on TRAIN last chain state
                G_zn_train = 1 ./ (1 + (zn_train ./ Kc).^nc);

                Theta_train   = [Theta_base, G_zn_train];
                termNames     = [baseNames, {'Hill(z_n)'}];

                % Normalize features 
                colscale              = vecnorm(Theta_train, 2, 1);
                colscale(colscale==0) = 1;
                ThetaN                = Theta_train ./ colscale;

                % Normalize derivative targets 
                if normalize_dX
                    dx_scale              = vecnorm(dx_train0, 2, 1);
                    dx_scale(dx_scale==0) = 1;
                    dxN                   = dx_train0 ./ dx_scale;
                else
                    dx_scale = [1,1];
                    dxN      = dx_train0;
                end

                % SINDy (STRidge) on TRAIN only
                tried = tried + 1;
                [XiN, ~, ~] = STRidge_sweep(ThetaN, dxN, ...
                                            ridge_list, lambda_thresh, max_iter_str);

                % Un-normalize Xi back to original scale
                Xi = (XiN ./ colscale.').* dx_scale;

                % Roll out identified model over FULL horizon
                par_roll = struct('polyorder',polyorder,'K',Kc,'nH',nc, ...
                                  'n_chain',n_chain,'r_chain',r_chain, ...
                                  'Xi',Xi);

                y0_roll = [M0; P0; repmat(P_s(1), n_chain, 1)];

                try
                    [tBraw, yBraw] = ode45(@(tt,yy) rollout_identified_model(tt,yy,par_roll), ...
                                           [0 t(end)], y0_roll, optsO);
                catch
                    continue;
                end

                x_id_full = interp1(tBraw, yBraw(:,1:2), t, 'linear');
                x_id_use  = x_id_full(idx_all,:);

                % TRAIN trajectory error  
                e_train = x_id_use(isTrain,:) - xref_use(isTrain,:);
                goodT   = all(isfinite(e_train),2);
                e_train = e_train(goodT,:);
                if isempty(e_train), continue; end
                mse_train = mean(e_train.^2,'all');

                %  VAL trajectory error 
                e_val = x_id_use(isVal,:) - xref_val;
                goodV = all(isfinite(e_val),2);
                e_val = e_val(goodV,:);
                if isempty(e_val), continue; end

                kept    = kept + 1;
                mse_val = mean(e_val.^2,'all');

                rmsM_val = sqrt(mean(e_val(:,1).^2));
                rmsP_val = sqrt(mean(e_val(:,2).^2));

                % Complexity, active terms across both equations
                k = nnz(abs(Xi) > thrK);

                % BIC-style score on valdation trajectory.
                Nval    = size(e_val,1);
                BIC_val = Nval*log(mse_val + 1e-12) + k*log(Nval) ...
                        + bic_penalty_per_chain*n_chain + bic_penalty_Kn;

                %  BIC on TRAIN trajectory for reporting
                Ntr       = size(e_train,1);
                BIC_train = Ntr*log(mse_train + 1e-12) + k*log(Ntr) ...
                          + bic_penalty_per_chain*n_chain + bic_penalty_Kn;

                cand(end+1) = struct( ... 
                    'n',n_chain,'tau',tau,'K',Kc,'nH',nc,'r',r_chain,'Xi',Xi, ...
                    'BIC_val',BIC_val,'mse_val',mse_val,'rmsM_val',rmsM_val,'rmsP_val',rmsP_val, ...
                    'BIC_train',BIC_train,'mse_train',mse_train,'k',k, ...
                    'termNames',{termNames});

                if isempty(best) || BIC_val < best.BIC_val
                    best = cand(end);
                end

            end
        end
    end
end

fprintf('Tried %d combos; kept %d.\n', tried, kept);
if isempty(best)
    error('No valid combos. Adjust grids or tolerances.');
end

%% 4) Show top models and BEST model (by VAL-BIC)
[~,ord] = sort([cand.BIC_val]);
cand    = cand(ord);

Kshow = min(5, numel(cand));
fprintf('\n=== Top %d candidates (by VAL-BIC) ===\n', Kshow);
for i = 1:Kshow
    c = cand(i);
    fprintf('%2d) n=%-3d  tau=%6.2f  r=%7.3f  K=%4d  nH=%2d | ', ...
        i, c.n, c.tau, c.r, c.K, c.nH);
    fprintf('RMS_val(M)=%.3g  RMS_val(P)=%.3g | BIC_val=%.2f | active=%d\n', ...
        c.rmsM_val, c.rmsP_val, c.BIC_val, c.k);
end

Xi        = best.Xi;
termNames = best.termNames;
thr_show  = 1e-6;

Xi_print  = Xi;
Xi_print(abs(Xi_print) < thr_show) = 0;

fprintf('\n=== BEST model (selected by VALIDATION trajectory) ===\n');
fprintf('n = %d, tau = %.6g, r = n/tau = %.6g, K = %d, nH = %d | ', ...
        best.n, best.tau, best.r, best.K, best.nH);
fprintf('BIC_val = %.2f | BIC_train = %.2f | MSE_train = %.3e | MSE_val = %.3e | active = %d\n', ...
        best.BIC_val, best.BIC_train, best.mse_train, best.mse_val, best.k);

try
    T = array2table(Xi_print, ...
        'RowNames',     termNames, ...
        'VariableNames',{'dMdt','dPdt'});
    disp('--- SINDy Coefficient Matrix Xi ---');
    disp(T);
catch
    disp('--- SINDy Coefficient Matrix Xi ---');
    disp(Xi_print);
end

fprintf('\nIdentified BEST model:\n');
fprintf('dM/dt = '); printSINDyEq(Xi(:,1), termNames);
fprintf('dP/dt = '); printSINDyEq(Xi(:,2), termNames); fprintf('\n');

%% 5) Final rollout using best model 
par_best = struct('polyorder',polyorder,'K',best.K,'nH',best.nH, ...
                  'n_chain',best.n,'r_chain',best.r, ...
                  'Xi',best.Xi);

y0_best = [M0; P0; repmat(P_s(1), best.n, 1)];
[tBbest, yBbest] = ode45(@(tt,yy) rollout_identified_model(tt,yy,par_best), ...
                         [0 t(end)], y0_best, optsO);

x_id_best = interp1(tBbest, yBbest(:,1:2), t, 'linear');

%% 6) Error metrics 
x_ref_final = x_ref_all;

xid_use   = x_id_best(idx_all,:);
xref_use2 = x_ref_final(idx_all,:);

e_train = xid_use(isTrain,:) - xref_use2(isTrain,:);
e_val   = xid_use(isVal,:)   - xref_use2(isVal,:);


e_train = e_train(all(isfinite(e_train),2),:);
e_val   = e_val(all(isfinite(e_val),2),:);

mse_train = mean(e_train.^2,'all');
mse_val   = mean(e_val.^2,'all');

rss_train = sum(e_train.^2,'all');
rss_val   = sum(e_val.^2,'all');

fprintf('\n--- TRAJECTORY METRICS ---\n');
fprintf('TRAIN: RSS = %.3e | MSE = %.3e | RMS(M)=%.3e | RMS(P)=%.3e\n', ...
    rss_train, mse_train, sqrt(mean(e_train(:,1).^2)), sqrt(mean(e_train(:,2).^2)));
fprintf('VAL  : RSS = %.3e | MSE = %.3e | RMS(M)=%.3e | RMS(P)=%.3e\n', ...
    rss_val, mse_val, sqrt(mean(e_val(:,1).^2)), sqrt(mean(e_val(:,2).^2)));

%% 7) FIGURE: Training data (noisy vs smoothed) 
figure('Color','w');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(t, M, '.', 'MarkerSize', 6); hold on;
plot(t, M_s, '-', 'LineWidth', 1.2);
xline(t_split, '--', 'LineWidth', 1.2);
grid on;
xlabel('Time (min)');
ylabel('M(t)');
legend('Measured M (noisy)', 'Smoothed M_s', 'Train/Val split', 'Location','best');

nexttile;
plot(t, P, '.', 'MarkerSize', 6); hold on;
plot(t, P_s, '-', 'LineWidth', 1.2);
xline(t_split, '--', 'LineWidth', 1.2);
grid on;
xlabel('Time (min)');
ylabel('P(t)');
legend('Measured P (noisy)', 'Smoothed P_s', 'Train/Val split', 'Location','best');
print(gcf, '-depsc', '-r300', 'training_data1.eps')
annotation('textbox', [0.02 0.02 0.05 0.05], ...
    'String','(c)', ...
    'FontSize',14, ...
    'FontWeight','bold', ...
    'LineStyle','none');

%% 8) FIGURE: Truth vs identified + split 
figure('Color','w');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

% Protein
nexttile;
plot(t, x_true(:,2), '-', 'LineWidth', 1.0); hold on;
plot(t, x_id_best(:,2), '--', 'LineWidth', 1.4);
xline(t_split, '--', 'LineWidth', 1.2);
grid on;
xlabel('Time (min)');
ylabel('P(t)');
legend('Truth (LCT)', 'Identified model', 'Train/Val split', 'Location','best');

% mRNA
nexttile;
plot(t, x_true(:,1), '-', 'LineWidth', 1.0); hold on;
plot(t, x_id_best(:,1), '--', 'LineWidth', 1.4);
xline(t_split, '--', 'LineWidth', 1.2);
grid on;
xlabel('Time (min)');
ylabel('M(t)');
legend('Truth (LCT)', 'Identified model', 'Train/Val split', 'Location','best');
print(gcf, '-depsc', '-r300', 'truth_vs_identified1.eps')
annotation('textbox', [0.02 0.02 0.05 0.05], ...
    'String','(d)', ...
    'FontSize',14, ...
    'FontWeight','bold', ...
    'LineStyle','none');


% END Hes1_mRNA_example.m

