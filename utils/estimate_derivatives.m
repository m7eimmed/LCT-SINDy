function [dMdt, dPdt, M_s, P_s] = estimate_derivatives(t, M, P, mode)
% mode: false/'raw' | 'sgolay' 
% Returns derivatives and the smoothed signals used.

if islogical(mode), mode = tern(mode,'sgolay','raw'); end
if nargin<4 || isempty(mode), mode = 'raw'; end

switch lower(mode)
    case {'raw','false'}
        M_s  = M;  P_s  = P;
        dMdt = gradient(M_s, t);
        dPdt = gradient(P_s, t);

    case 'sgolay'
        % Savitzky–Golay smoothing, window adapted to sampling
        dt = median(diff(t));
        win = max(5, round(30/dt));  % ~30 time-units wide
        if mod(win,2)==0, win = win+1; end
        deg = min(3, win-2);
        M_s  = sgolayfilt(M, deg, win);
        P_s  = sgolayfilt(P, deg, win);
        dMdt = gradient(M_s, t);
        dPdt = gradient(P_s, t);


    otherwise
        error('Unknown mode for estimate_derivatives: %s', mode);
end
end

function y = tern(cond,a,b)
if cond, y=a; else, y=b; end
end


