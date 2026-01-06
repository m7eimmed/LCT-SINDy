function dydt = rhs_dist_Hes1_LCT(y, Mm, Mp, P0, N, alpha_m, alpha_p, a, p)
% Ground-truth distributed-delay Hes1 model implemented via LCT chain.
% y = [M; P; z1..zp], Hill feedback uses z_p.

M = y(1); 
P = y(2);
z = y(3:end);
zp = z(end);

G     = 1 / (1 + (zp / P0)^N);
dMdt  = alpha_m * G     - Mm * M;
dPdt  = alpha_p * M     - Mp * P;

dz        = zeros(p,1);
dz(1)     = a*(P - z(1));
for k = 2:p
    dz(k) = a*(z(k-1) - z(k));
end

dydt = [dMdt; dPdt; dz];
end
