function dy = rollout_logistic_LCT_SINDy(~, y, par)
%ROLLOUT_LOGISTIC_LCT_SINDY  RHS for identified distributed-delay logistic model.
%
% States:
%   y = [x; z1; ...; zn]
%
% par fields:
%   polyorder : integer >= 1  (total degree in (x, z_n))
%   n_chain   : length of LCT chain
%   a_chain   : Erlang rate (n_chain / tau)
%   Xi        : SINDy coefficient vector for dx/dt, size [m x 1]
%
% IMPORTANT:
%   The feature row must match build_logistic_library(x,z,polyorder):
%   monomials x^i z^j with i+j <= polyorder, ordered by total degree d,
%   then i = 0..d (so j = d-i).

    % unpack states
    x  = y(1);
    zc = y(2:end);
    zn = zc(end);             % terminal chain state
    n  = par.n_chain;
    a  = par.a_chain;

    % Build feature row in same order as build_logistic_library
    p = par.polyorder;
    m = (p+1)*(p+2)/2;        % number of monomials with i+j<=p
    row = zeros(1, m);

    idx = 1;
    for d = 0:p
        for i = 0:d
            j = d - i;
            row(idx) = (x^i) * (zn^j);
            idx = idx + 1;
        end
    end

    % dx/dt from SINDy coefficients (ensure Xi is a column)
    Xi = par.Xi(:);
    dxdt = row * Xi;

    % LCT chain dynamics driven by x(t)
    dz = zeros(n,1);
    dz(1) = a * (x - zc(1));
    for k = 2:n
        dz(k) = a * (zc(k-1) - zc(k));
    end

    % Pack RHS
    dy = [dxdt; dz];
end
