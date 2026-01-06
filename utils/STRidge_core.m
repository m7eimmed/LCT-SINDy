function Xi = STRidge_core(Theta, dX, gamma, lam, maxit)
% STRidge_core:
%   1. ridge fit
%   2. threshold small coeffs
%   3. refit only on active terms
% repeat.

[~, m] = size(Theta);
p      = size(dX,2);

G  = Theta.'*Theta + gamma*eye(m);
Xi = G \ (Theta.'*dX);

for it = 1:maxit 
    small = abs(Xi) < lam;         
    Xi(small) = 0;

    for j = 1:p
        bigj = ~small(:,j);
        if any(bigj)
            Tj = Theta(:,bigj);
            Xi(bigj,j) = (Tj.'*Tj + gamma*eye(sum(bigj))) \ (Tj.'*dX(:,j));
        else
            Xi(:,j) = 0;
        end
    end
end
end
