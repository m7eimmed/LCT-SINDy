function dydt = rollout_identified_model(~, y, par)
%ROLLOUT_IDENTIFIED_MODEL  RHS for identified Hes1 LCT-SINDy model.
%
% State:
%   y = [M; P; z1; ...; zn]
%
% Model:
%   [dM; dP]/dt = Theta(M,P,zn) * Xi
%   where Theta = [poly_lib(M,P), Hill(zn)]
%
% LCT chain:
%   z1' = r (P - z1)
%   zi' = r (z_{i-1} - zi), i=2..n
%
% Required par fields:
%   polyorder, K, nH, n_chain, r_chain, Xi

M = y(1);
P = y(2);
z = y(3:end);
zn = z(end);

% Build feature row (M,P polynomial library) 
row = 1; % constant term
for k = 1:par.polyorder
    for a = k:-1:0
        b = k - a;
        row = [row, (M^a)*(P^b)]; 
    end
end

% Hill repression using terminal chain state zn 
Gzn = 1 / (1 + (zn / par.K)^par.nH);
row = [row, Gzn]; 

% Identified dynamics for [M; P] 
dMP = (row * par.Xi).';   

% LCT chain dynamics 
n  = par.n_chain;
r  = par.r_chain;

dz = zeros(n,1);
dz(1) = r*(P - z(1));
for i = 2:n
    dz(i) = r*(z(i-1) - z(i));
end

dydt = [dMP; dz];
end

