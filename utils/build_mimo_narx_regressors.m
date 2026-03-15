function [X, Y, idxMap] = build_mimo_narx_regressors(y, u, naMat, nbMat, nkMat)

    [N, ny] = size(y);
    nu = size(u,2);

    maxLagY = max(naMat(:));
    maxLagU = max(nbMat(:) + nkMat(:) - 1);
    maxLag  = max(maxLagY, maxLagU);

    % Universal regressor:
    % [all past y's up to maxLagY, all past u's up to maxLagU]
    D = ny*maxLagY + nu*maxLagU;

    X = zeros(N-maxLag, D);
    Y = zeros(N-maxLag, ny);
    idxMap = zeros(N-maxLag,1);

    row = 0;
    for k = maxLag+1:N
        row = row + 1;
        phi = [];

        % past outputs
        for j = 1:ny
            phi = [phi, y(k-1:-1:k-maxLagY, j).']; %#ok<AGROW>
        end

        % past inputs
        for m = 1:nu
            phi = [phi, u(k-1:-1:k-maxLagU, m).']; %#ok<AGROW>
        end

        X(row,:) = phi;
        Y(row,:) = y(k,:);
        idxMap(row) = k;
    end
end