function [X, Y, idxMap] = build_narx_regressors(y, u, na, nb, nk)

    N = length(y);
    maxLag = max(na, nb + nk - 1);

    X = zeros(N - maxLag, na + nb);
    Y = zeros(N - maxLag, 1);
    idxMap = zeros(N - maxLag, 1);

    row = 0;
    for k = maxLag+1:N
        row = row + 1;

        phiY = y(k-1:-1:k-na).';
        phiU = u(k-nk:-1:k-nk-nb+1).';

        X(row,:) = [phiY, phiU];
        Y(row) = y(k);
        idxMap(row) = k;
    end
end