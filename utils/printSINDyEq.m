function printSINDyEq(c, names)


thr   = 1e-8; 
first = true;
for i = 1:numel(c)
    ci = c(i);
    if abs(ci) > thr
        if first
            if ci < 0, fprintf('-'); end
            first = false;
        else
            if ci>=0, fprintf(' + '); else, fprintf(' - '); end
        end
        fprintf('%.6g*%s', abs(ci), names{i});
    end
end

if first
    fprintf('0');
end
fprintf('\n');
end
