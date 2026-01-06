function dy = rollout_ikeda_LCT_SINDy(~, y, par)
%ROLLOUT_IKEDA_LCT_SINDY  ODE rollout for identified LCT-SINDy Ikeda model.
%
% States:
%   y = [x; z1; ...; z_p]
%
% par fields:
%   polyorder          : polynomial order in (x,z_p)
%   include_cross_sin  : include x*sin(z_p), z_p*sin(z_p)
%   p_chain            : chain length p
%   a_chain            : rate a = p/tau
%   Xi                 : coefficient vector

    x  = y(1);
    zc = y(2:end);
    zp = zc(end);

    p = par.p_chain;
    a = par.a_chain;

    % Build feature row in the SAME order as build_ikeda_library:
    % 1) monomials by total degree d, i=0..d
    % 2) sin(z)
    % 3) optional x*sin(z), z*sin(z)
    polyorder = par.polyorder;

    % number of monomials i+j<=polyorder:
    m_poly = (polyorder+1)*(polyorder+2)/2;

    row = zeros(1, m_poly + 1 + (2 * par.include_cross_sin));
    idx = 1;

    for d = 0:polyorder
        for i = 0:d
            j = d - i;
            row(idx) = (x^i) * (zp^j);
            idx = idx + 1;
        end
    end

    s = sin(zp);
    row(idx) = s; idx = idx + 1;

    if par.include_cross_sin
        row(idx) = x*s; idx = idx + 1;
        row(idx) = zp*s; idx = idx + 1;
    end

    Xi = par.Xi(:);
    dxdt = row * Xi;

    % LCT chain driven by x(t)
    dz = zeros(p,1);
    dz(1) = a * (x - zc(1));
    for k = 2:p
        dz(k) = a * (zc(k-1) - zc(k));
    end

    dy = [dxdt; dz];
end
