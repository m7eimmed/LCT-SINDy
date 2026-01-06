function [Xi, gamma_best, crit_best] = STRidge_sweep(ThetaN, dXN, ridge_list, lam, maxit)
% Try multiple ridge strengths, run STRidge_core, and pick the best by
% residual + tiny sparsity penalty.

bestXi     = [];
gamma_best = ridge_list(1);
crit_best  = inf;

for gamma = ridge_list
    Xi_try = STRidge_core(ThetaN, dXN, gamma, lam, maxit);
    resid  = dXN - ThetaN*Xi_try;
    mse_dx = mean(resid.^2, 'all');
    k_act  = nnz(abs(Xi_try) > 0);
    crit   = mse_dx + 1e-8*k_act;
    if crit < crit_best
        crit_best  = crit;
        gamma_best = gamma;
        bestXi     = Xi_try;
    end
end

Xi = bestXi;
end
