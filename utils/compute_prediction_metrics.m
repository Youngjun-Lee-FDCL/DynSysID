function metrics = compute_prediction_metrics(yTrue, yPred, ySim)
% Compute RMSE and FIT metrics for one-step and free-run results.

    ny = size(yTrue, 2);

    rmse1_each    = sqrt(mean((yTrue - yPred).^2, 1));
    rmseFree_each = sqrt(mean((yTrue - ySim ).^2, 1));

    fit1_each    = zeros(1, ny);
    fitFree_each = zeros(1, ny);

    for j = 1:ny
        denom = norm(yTrue(:,j) - mean(yTrue(:,j)));
        if denom < 1e-12
            fit1_each(j)    = NaN;
            fitFree_each(j) = NaN;
        else
            fit1_each(j)    = 100 * (1 - norm(yTrue(:,j) - yPred(:,j)) / denom);
            fitFree_each(j) = 100 * (1 - norm(yTrue(:,j) - ySim(:,j))  / denom);
        end
    end

    metrics = struct();
    metrics.rmse1_each    = rmse1_each;
    metrics.rmseFree_each = rmseFree_each;
    metrics.fit1_each     = fit1_each;
    metrics.fitFree_each  = fitFree_each;

    metrics.rmse1    = mean(rmse1_each, 'omitnan');
    metrics.rmseFree = mean(rmseFree_each, 'omitnan');
    metrics.fit1     = mean(fit1_each, 'omitnan');
    metrics.fitFree  = mean(fitFree_each, 'omitnan');
end