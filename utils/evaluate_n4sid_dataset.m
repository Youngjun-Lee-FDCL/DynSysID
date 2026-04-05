function result = evaluate_n4sid_dataset( ...
    dataEval, evalTag, evalMode, modelName, ...
    sysN4, Ts)

    [~, y0] = force_2d_io(dataEval.uCell{1}, dataEval.yCell{1});
    ny = size(y0,2);

    rmse1_all         = zeros(dataEval.numExp,1);
    fit1_all          = zeros(dataEval.numExp,1);
    rmseFree_all      = zeros(dataEval.numExp,1);
    fitFree_all       = zeros(dataEval.numExp,1);

    rmse1_each_all    = zeros(dataEval.numExp, ny);
    fit1_each_all     = zeros(dataEval.numExp, ny);
    rmseFree_each_all = zeros(dataEval.numExp, ny);
    fitFree_each_all  = zeros(dataEval.numExp, ny);

    freeRunTime_all   = zeros(dataEval.numExp,1);

    yTrueCell = cell(dataEval.numExp,1);
    yPredCell = cell(dataEval.numExp,1);
    ySimCell  = cell(dataEval.numExp,1);
    tEvalCell = cell(dataEval.numExp,1);

    for eTest = 1:dataEval.numExp
        fprintf('\n==================================================\n');
        fprintf('Evaluation tag : %s\n', evalTag);
        fprintf('Testing experiment %d / %d\n', eTest, dataEval.numExp);
        fprintf('==================================================\n');

        expResult = evaluate_single_experiment_n4sid( ...
            dataEval, eTest, evalTag, evalMode, modelName, ...
            sysN4, Ts);

        rmse1_all(eTest)         = expResult.metrics.rmse1;
        fit1_all(eTest)          = expResult.metrics.fit1;
        rmseFree_all(eTest)      = expResult.metrics.rmseFree;
        fitFree_all(eTest)       = expResult.metrics.fitFree;

        rmse1_each_all(eTest,:)    = expResult.metrics.rmse1_each(:).';
        fit1_each_all(eTest,:)     = expResult.metrics.fit1_each(:).';
        rmseFree_each_all(eTest,:) = expResult.metrics.rmseFree_each(:).';
        fitFree_each_all(eTest,:)  = expResult.metrics.fitFree_each(:).';

        freeRunTime_all(eTest)   = expResult.freeRunTime;

        yTrueCell{eTest} = expResult.yTrue;
        yPredCell{eTest} = expResult.yPred;
        ySimCell{eTest}  = expResult.ySim;
        tEvalCell{eTest} = expResult.tEval;
    end

    result = struct();
    result.rmse1_all = rmse1_all;
    result.fit1_all = fit1_all;
    result.rmseFree_all = rmseFree_all;
    result.fitFree_all = fitFree_all;

    result.rmse1_each_all = rmse1_each_all;
    result.fit1_each_all = fit1_each_all;
    result.rmseFree_each_all = rmseFree_each_all;
    result.fitFree_each_all = fitFree_each_all;

    result.freeRunTime_all = freeRunTime_all;

    result.rmse1    = mean(rmse1_all);
    result.fit1     = mean(fit1_all);
    result.rmseFree = mean(rmseFree_all);
    result.fitFree  = mean(fitFree_all);

    result.yTrueCell = yTrueCell;
    result.yPredCell = yPredCell;
    result.ySimCell  = ySimCell;
    result.tEvalCell = tEvalCell;

    fprintf('\n==================================================\n');
    fprintf('Average performance over evaluation set: %s\n', evalTag);
    fprintf('==================================================\n');
    fprintf('Average one-step RMSE : %.6f\n', result.rmse1);
    fprintf('Average one-step FIT  : %.6f\n', result.fit1);
    fprintf('Average free-run RMSE : %.6f\n', result.rmseFree);
    fprintf('Average free-run FIT  : %.6f\n', result.fitFree);
    fprintf('Average free-run time : %.6f seconds\n', mean(freeRunTime_all));
end