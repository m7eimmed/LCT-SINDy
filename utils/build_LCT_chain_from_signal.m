function zn_full = build_LCT_chain_from_signal(t, x, n_chain, tau, optsZ)
%BUILD_LCT_CHAIN_FROM_SIGNAL  LCT (Erlang) chain driven by a scalar signal.
%
%   zn_full = build_LCT_chain_from_signal(t, x, n_chain, tau, optsZ)
%
% Inputs:
%   t        : time grid (vector, length N)
%   x        : scalar signal x(t) on the same grid (vector, length N)
%   n_chain  : length of the LCT chain (integer p >= 1)
%   tau      : mean delay ( > 0 ), Erlang rate a = n_chain / tau
%   optsZ    : odeset options for ode45 
%
% Output:
%   zn_full  : terminal chain state z_n(t) evaluated on the original t-grid
%
% Dynamics:
%   z_1'(t) = a ( x(t)   - z_1(t) )
%   z_j'(t) = a ( z_{j-1}(t) - z_j(t) ),  j = 2,...,n_chain
%
% with a = n_chain / tau and initial condition z_j(0) = x(0).

    t = t(:);
    x = x(:);

    if nargin < 5 || isempty(optsZ)
        optsZ = odeset('RelTol',1e-9,'AbsTol',1e-11);
    end

    a_chain = n_chain / tau;

    % Interpolant for the driving signal x(t)
    x_of_t = @(tt) interp1(t, x, tt, 'pchip', 'extrap');

    % Initial chain state: all nodes start at x(0)
    z0 = repmat(x(1), n_chain, 1);

    % ODE RHS for the LCT chain
    odeZ = @(tt,zz) lct_chain_rhs_scalar(tt, zz, x_of_t, a_chain);

    % Integrate the chain over [t(1), t(end)]
    [tZ, zAll] = ode45(odeZ, [t(1) t(end)], z0, optsZ);

    % Interpolate terminal state z_n back to the original t-grid
    zAll_interp = interp1(tZ, zAll, t, 'pchip');
    zn_full     = zAll_interp(:, end);
end

% -------------------------------------------------------------------------
function dz = lct_chain_rhs_scalar(tt, z, x_of_t, a)
% helper for build_LCT_chain_from_signal

    xval = x_of_t(tt);
    n    = numel(z);

    dz      = zeros(n,1);
    dz(1)   = a * (xval - z(1));
    for k = 2:n
        dz(k) = a * (z(k-1) - z(k));
    end
end

