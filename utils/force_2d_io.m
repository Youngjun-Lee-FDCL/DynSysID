function [u, y] = force_2d_io(uRaw, yRaw)
    u = uRaw;
    y = yRaw;

    if isvector(u), u = u(:); end
    if isvector(y), y = y(:); end
end