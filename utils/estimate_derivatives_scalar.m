function [dxdt, x_s] = estimate_derivatives_scalar(t, x, mode)
%ESTIMATE_DERIVATIVES_SCALAR  Derivative + smoothing for a single signal.
%
%   [dxdt, x_s] = estimate_derivatives_scalar(t, x, mode)
%
% mode: 'raw' | 'sgolay' 
%   'raw'   : no smoothing, simple gradient
%   'sgolay': Savitzky–Golay smoothing + gradient
%
% Outputs:
%   dxdt : estimated derivative dx/dt on the same grid t
%   x_s  : smoothed signal used to build the LCT chain (for 'raw'/'tv'
%          we just return x; for 'sgolay' we return the sgolay-smoothed x).

    if nargin < 3 || isempty(mode)
        mode = 'raw';
    end

    
    if islogical(mode)
        if mode
            mode = 'sgolay';
        else
            mode = 'raw';
        end
    end

    t = t(:);
    x = x(:);

    switch lower(mode)
        case {'raw','false'}
            x_s  = x;
            dxdt = gradient(x_s, t);

        case 'sgolay'
            % Savitzky–Golay smoothing, window adapted to sampling
            dt = median(diff(t));
            win = max(5, round(2.5/dt));  % Use a window size of 0.5/dt for the Ikeda example and 2.5 for the logistic example.
            if mod(win,2)==0, win = win+1; end
            deg = min(3, win-2);
            x_s  = sgolayfilt(x, deg, win);
            dxdt = gradient(x_s, t);

      

        otherwise
            error('Unknown mode for estimate_derivatives_scalar: %s', mode);
    end
end


