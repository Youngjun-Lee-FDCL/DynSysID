function expResult = evaluate_single_experiment_n4sid( ...
    dataEval, eTest, evalTag, evalMode, modelName, ...
    sysN4, Ts)

    %#ok<INUSD>
    switch lower(evalMode)
        case 'val'
            [uEval, yTrue] = force_2d_io(dataEval.uValCell{eTest}, dataEval.yValCell{eTest});

            if isfield(dataEval, 'tCell') && isfield(dataEval, 'idxValCell')
                tEval = dataEval.tCell{eTest}(dataEval.idxValCell{eTest});
            else
                tEval = (0:size(yTrue,1)-1)' * Ts;
            end

        case 'full'
            [uEval, yTrue] = force_2d_io(dataEval.uCell{eTest}, dataEval.yCell{eTest});

            if isfield(dataEval, 'tCell')
                tEval = dataEval.tCell{eTest};
            else
                tEval = (0:size(yTrue,1)-1)' * Ts;
            end

        otherwise
            error('evalMode must be ''full'' or ''val''.');
    end

    zEval = iddata(yTrue, uEval, Ts);

    % one-step prediction
    yPredObj = predict(sysN4, zEval, 1);
    yPred = yPredObj.OutputData;

    % free-run simulation
    tic;
    ySimObj = sim(sysN4, zEval);
    freeRunTime = toc;
    ySim = ySimObj.OutputData;

    metrics = compute_prediction_metrics(yTrue, yPred, ySim);

    fprintf('One-step RMSE : %.6f\n', metrics.rmse1);
    fprintf('One-step FIT  : %.6f\n', metrics.fit1);
    fprintf('Free-run RMSE : %.6f\n', metrics.rmseFree);
    fprintf('Free-run FIT  : %.6f\n', metrics.fitFree);
    fprintf('Free-run time : %.6f seconds\n', freeRunTime);

    expResult = struct();
    expResult.metrics = metrics;
    expResult.freeRunTime = freeRunTime;

    expResult.yTrue = yTrue;
    expResult.yPred = yPred;
    expResult.ySim  = ySim;
    expResult.tEval = tEval;
end