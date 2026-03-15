function xNow = build_single_mimo_regressor_for_rollout( ...
    kOrig, testStartOriginalIndex, yMeasured, uMeasured, yHatFree, ...
    naMat, nbMat, nkMat)

    ny = size(yMeasured,2);
    nu = size(uMeasured,2);

    maxLagY = max(naMat(:));
    maxLagU = max(nbMat(:) + nkMat(:) - 1);

    phi = [];

    % past outputs: use measured outputs before validation start,
    % and predicted outputs after validation start
    for j = 1:ny
        vec = zeros(1, maxLagY);
        for ell = 1:maxLagY
            pastIdx = kOrig - ell;
            if pastIdx < testStartOriginalIndex
                vec(ell) = yMeasured(pastIdx, j);
            else
                vec(ell) = yHatFree(pastIdx - testStartOriginalIndex + 1, j);
            end
        end
        phi = [phi, vec]; %#ok<AGROW>
    end

    % past inputs: always known
    for m = 1:nu
        vec = uMeasured(kOrig-1:-1:kOrig-maxLagU, m).';
        phi = [phi, vec]; %#ok<AGROW>
    end

    xNow = phi;
end