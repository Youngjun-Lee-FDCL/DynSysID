function result = evaluate_n4sid_dataset_with_plots(data, datasetName, evalMode, modelName, sys, Ts)

    numExp = data.numExp;

    yTrueCell = cell(numExp,1);
    yPredCell = cell(numExp,1);
    ySimCell  = cell(numExp,1);
    tCell     = cell(numExp,1);

    for e = 1:numExp
        switch lower(evalMode)
            case 'val'
                [uEval, yEval] = force_2d_io(data.uValCell{e}, data.yValCell{e});

                if isfield(data, 'tCell') && isfield(data, 'idxValCell')
                    tEval = data.tCell{e}(data.idxValCell{e});
                else
                    tEval = (0:size(yEval,1)-1)' * Ts;
                end

            case 'full'
                [uEval, yEval] = force_2d_io(data.uCell{e}, data.yCell{e});

                if isfield(data, 'tCell')
                    tEval = data.tCell{e};
                else
                    tEval = (0:size(yEval,1)-1)' * Ts;
                end

            otherwise
                error('evalMode must be ''full'' or ''val''.');
        end

        zEval = iddata(yEval, uEval, Ts);

        % 초기상태 추정
        x0est = findstates(sys, zEval, Inf);

        simOpt  = simOptions('InitialCondition', x0est);
        predOpt = predictOptions('InitialCondition', x0est);

        yPred = predict(sys, zEval, 1, predOpt).OutputData;
        ySim  = sim(sys, zEval, simOpt).OutputData;

        yTrueCell{e} = yEval;
        yPredCell{e} = yPred;
        ySimCell{e}  = ySim;
        tCell{e}     = tEval;
    end

    % 전체 metric 계산용 concat
    yTrueAll = vertcat(yTrueCell{:});
    yPredAll = vertcat(yPredCell{:});
    ySimAll  = vertcat(ySimCell{:});

    result = compute_prediction_metrics(yTrueAll, yPredAll, ySimAll);
    result.yTrueCell = yTrueCell;
    result.yPredCell = yPredCell;
    result.ySimCell  = ySimCell;
    result.tCell     = tCell;

    % plot
    plot_pred_sim_per_experiment( ...
        tCell, yTrueCell, yPredCell, ySimCell, ...
        sprintf('%s | %s', modelName, datasetName));

    plot_residual_per_experiment( ...
        tCell, yTrueCell, yPredCell, ySimCell, ...
        sprintf('%s | %s', modelName, datasetName));
end