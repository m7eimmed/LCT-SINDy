function [Theta, names] = build_ikeda_library(x, z, polyorder, include_cross_sin)
%BUILD_IKEDA_LIBRARY  Polynomial library in (x,z) up to total degree polyorder,
% plus sin(z) and optional cross terms with sin(z).
%
% Base : all monomials x^i z^j with i+j <= polyorder
% Extra: sin(z)
% Optional: x*sin(z), z*sin(z)

    if nargin < 4, include_cross_sin = true; end

    x = x(:); z = z(:);
    N = numel(x);

    Theta = [];
    names = {};

    % Polynomial part 
    for d = 0:polyorder
        for i = 0:d
            j = d - i;
            term = (x.^i) .* (z.^j);
            Theta = [Theta, term]; 

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

    % Sine feature(s)
    s = sin(z);
    Theta = [Theta, s]; 
    names{end+1} = 'sin(z)';

    if include_cross_sin
        Theta = [Theta, x.*s, z.*s]; 
        names{end+1} = 'x*sin(z)';
        names{end+1} = 'z*sin(z)';
    end

    % quick sanity
    if size(Theta,1) ~= N
        error('Library size mismatch.');
    end
end
