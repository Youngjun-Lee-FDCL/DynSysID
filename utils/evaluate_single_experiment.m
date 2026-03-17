function expResult = evaluate_single_experiment( ...
    dataEval, eTest, evalTag, evalMode, modelName, ...
    naMat, nbMat, nkMat, ...
    gpModel, normStat, opt)

    t    = dataEval.tCell{eTest};
    [uRaw, yRaw] = force_2d_io(dataEval.uCell{eTest}, dataEval.yCell{eTest});

    N  = size(yRaw,1);
    ny = size(yRaw,2);

    u = uRaw;
    y = yRaw;

    if opt.usePreprocessing
        frameNow = min(opt.sgolayFrame_y, N);
        if mod(frameNow,2) == 0
            frameNow = frameNow - 1;
        end
        if frameNow < 3
            error('Savitzky-Golay frame length for output is too short.');
        end

        for j = 1:ny
            y(:,j) = smoothdata(yRaw(:,j), 'sgolay', frameNow);
        end
    end

    [Xall, ~, idxAll] = build_mimo_narx_regressors(y, u, naMat, nbMat, nkMat);

    switch lower(evalMode)
        case 'val'
            idxEvalRaw = dataEval.idxValCell{eTest};
            evalMask = ismember(idxAll, idxEvalRaw);
        case 'full'
            evalMask = true(size(idxAll));
        otherwise
            error('Unknown evalMode: %s', evalMode);
    end

    Xte   = Xall(evalMask,:);
    idxTe = idxAll(evalMask);

    if isempty(idxTe)
        error('Evaluation set is empty for experiment %d.', eTest);
    end

    YteRaw = yRaw(idxTe,:);
    XteN   = apply_normalization(Xte, normStat.muX, normStat.stdX);

    % One-step
    muTeN = zeros(size(XteN,1), ny);
    s2TeN = zeros(size(XteN,1), ny);

    for j = 1:ny
        [muTeN(:,j), s2TeN(:,j)] = gpml_predict_mean_var_exact( ...
            gpModel.hypOpt{j}, gpModel.meanfunc, gpModel.covfunc, ...
            normStat.XtrN, XteN, gpModel.postCell{j});
    end

    muTe = muTeN .* normStat.stdY + normStat.muY;
    s2Te = s2TeN .* (normStat.stdY.^2);

    % Free-run
    tStart = tic;
    yHatFree = nan(size(YteRaw));
    testStartOriginalIndex = idxTe(1);

    for n = 1:size(YteRaw,1)
        kOrig = idxTe(n);

        xNow = build_single_mimo_regressor_for_rollout( ...
            kOrig, testStartOriginalIndex, y, u, yHatFree, ...
            naMat, nbMat, nkMat);

        xNowN = apply_normalization(xNow, normStat.muX, normStat.stdX);

        for j = 1:ny
            muNowN = gpml_predict_mean_exact_single( ...
                gpModel.hypOpt{j}, gpModel.meanfunc, gpModel.covfunc, ...
                normStat.XtrN, xNowN, gpModel.postCell{j});

            yHatFree(n,j) = muNowN * normStat.stdY(j) + normStat.muY(j);
        end
    end

    freeRunTime = toc(tStart);

    metrics = compute_prediction_metrics(YteRaw, muTe, yHatFree);

    fprintf('[%s] Exp %d one-step RMSE : %.6f\n', evalTag, eTest, metrics.rmse1);
    fprintf('[%s] Exp %d one-step FIT  : %.6f\n', evalTag, eTest, metrics.fit1);
    fprintf('[%s] Exp %d free-run RMSE : %.6f\n', evalTag, eTest, metrics.rmseFree);
    fprintf('[%s] Exp %d free-run FIT  : %.6f\n', evalTag, eTest, metrics.fitFree);
    fprintf('Free-run simulation runtime (exp %d): %.6f seconds\n', eTest, freeRunTime);

    algoName = sprintf('GP-NARX | %s | %s | Exp %d', modelName, evalTag, eTest);

    plot_measured_outputs(t, yRaw, sprintf('%s - %s - Exp %d', modelName, evalTag, eTest));

    plot_gp_prediction_with_band( ...
        t(idxTe), YteRaw, muTe, s2Te, ...
        metrics.rmse1_each, metrics.fit1_each, ...
        algoName);

    plot_estimation_result( ...
        t(idxTe), YteRaw, yHatFree, ...
        metrics.rmseFree_each, metrics.fitFree_each, ...
        algoName, ...
        'free-run', 'b--');

    plot_estimation_errors( ...
        t(idxTe), YteRaw, muTe, yHatFree, ...
        algoName);

    expResult = struct();
    expResult.metrics = metrics;
    expResult.muTe = muTe;
    expResult.s2Te = s2Te;
    expResult.yHatFree = yHatFree;
    expResult.YteRaw = YteRaw;
    expResult.idxTe = idxTe;
    expResult.freeRunTime = freeRunTime;
end