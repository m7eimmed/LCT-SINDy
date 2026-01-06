function [Theta, names] = build_poly_library(X, d)
% Build polynomial library Theta(M,P) up to total degree d.
% Columns: [1, M, P, M^2, M*P, P^2, ...]
% Also returns 'names' for each column.

M = X(:,1);  
P = X(:,2);

Theta = ones(size(X,1),1);  % degree 0
names = {'1'};

for k = 1:d  % total degree = k
    for a = k:-1:0
        b = k - a;
        Theta = [Theta, (M.^a).*(P.^b)]; 

        if a>0 && b>0
            names{end+1} = sprintf('M^%d*P^%d', a, b); 
        elseif a>0
            if a==1
                names{end+1} = 'M';
            else
                names{end+1} = sprintf('M^%d', a);
            end
        elseif b>0
            if b==1
                names{end+1} = 'P';
            else
                names{end+1} = sprintf('P^%d', b);
            end
        else
            names{end+1} = '1';
        end
    end
end
end
