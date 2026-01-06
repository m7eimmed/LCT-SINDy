function zn_full = build_LCT_chain_from_P(t, P, n_chain, tau, optsZ)
% For a candidate (n_chain, tau), build an LCT chain z1..z_n driven by P(t).
% Returns the last node z_n evaluated on the same t-grid.

r_chain = n_chain / tau;

P_of_t = @(tt) interp1(t, P, tt, 'pchip', 'extrap');
z0     = repmat(P(1), n_chain, 1);

odeZ = @(tt,zz) lct_chain_rhs_local(tt, zz, P_of_t, r_chain);

[tZ, zAll] = ode45(odeZ, [t(1) t(end)], z0, optsZ);

% interpolate z_n back on the original t-grid
zAll_interp = interp1(tZ, zAll, t, 'pchip');
zn_full     = zAll_interp(:,end);

end

function dz = lct_chain_rhs_local(tt, z, P_of_t, r)
% internal helper for build_LCT_chain_from_P
Pval = P_of_t(tt);
n    = numel(z);
dz   = zeros(n,1);
dz(1)= r*(Pval - z(1));
for k = 2:n
    dz(k) = r*(z(k-1) - z(k));
end
end
