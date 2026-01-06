function [Theta, names] = build_logistic_library(x, z, polyorder)
% Build polynomial library in (x,z) up to total degree = polyorder
%
% Includes all monomials:
%   x^i z^j  with i+j <= polyorder
%
% Inputs:
%   x, z      : column vectors 
%   polyorder : integer >= 1
%
% Outputs:
%   Theta : N-by-m matrix of features
%   names : 1-by-m cell array of term names

x = x(:);
z = z(:);
N = numel(x);

Theta = [];
names = {};

% loop over total degree
for d = 0:polyorder
    for i = 0:d
        j = d - i;

        % monomial x^i z^j
        term = (x.^i) .* (z.^j);
        Theta = [Theta, term]; 

        % name
        if i == 0 && j == 0
            names{end+1} = '1';
        elseif i == 0
            names{end+1} = sprintf('z^%d', j);
        elseif j == 0
            names{end+1} = sprintf('x^%d', i);
        else
            names{end+1} = sprintf('x^%d*z^%d', i, j);
        end
    end
end
end
