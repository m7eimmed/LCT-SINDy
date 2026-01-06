function dydt = rhs_dist_logistic_LCT(y, r, K, a, p)
% RHS for distributed-delay logistic model using an Erlang LCT chain.
%
% States:
%   y = [x; z1; z2; ...; zp]
%   x'(t)  = r x(t) (1 - z_p(t)/K)
%   z_1'   = a (x - z_1)
%   z_j'   = a (z_{j-1} - z_j),  j = 2,...,p
%
% Inputs:
%   y : column vector [x; z_1; ...; z_p]
%   r : intrinsic growth rate
%   K : carrying capacity
%   a : Erlang/LCT rate
%   p : (nominal) chain length (only used for documentation; the actual
%       length is inferred from y to avoid mismatch).

x = y(1);
z = y(2:end);

% infer chain length from state vector to avoid mismatches
p = numel(z);

% Logistic equation with distributed delay (terminal chain state)
x_dot = r * x * (1 - z(end)/K);

% LCT chain dynamics
dz      = zeros(p,1);
dz(1)   = a * (x - z(1));
for j = 2:p
    dz(j) = a * (z(j-1) - z(j));
end

dydt = [x_dot; dz];
end


