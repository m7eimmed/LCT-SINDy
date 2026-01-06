function dxdt = rhs_ikeda_dde(~, x, xlag, alpha)
%RHS_IKEDA_DDE  Discrete-delay Ikeda-type scalar DDE:
%   x'(t) = -x(t) + alpha * sin(x(t - tau))
%
% dde23 signature: f(t, x(t), Z), where Z(:,k) = x(t - lag_k)

    xtau = xlag(1);     % single lag
    dxdt = -x + alpha * sin(xtau);
end
