clear; clc; close all;

%% Add path
rootPath = genpath('./');
addpath(rootPath);

%% =========================
% User-editable settings
%% =========================
rng(1);

% Example generator
% exampleFcn = @generate_nonlinear_msd_example;
% exampleFcn = @generate_frigola_benchmark_example;
% exampleFcn = @generate_linear_msd_example;
% exampleFcn = @generate_toy_nonlinear_example;
% exampleFcn = @generate_nonlinear_twotank_example;
% exampleFcn = @generate_linear_mimo3_example;
exampleFcn = @generate_nonlinear_cstr_PID_example;
% exampleFcn = @generate_nonlinear_aircraft43_PID_example;
% exampleFcn = @generate_vgpssm_easy_example;

% Dataset sizes
numExp       = 1;    % training / held-out validation experiments
numSimRunExp = 3;    % independent simulation-run experiments

% Independent test option
useIndependentSimRunTest = true;
independentEvalMode      = 'full';   % 'full' or 'val'
independentSeed          = 100;

% N4SID setup
orders = 1:8;

optN4 = n4sidOptions;
optN4.Focus        = 'simulation';
optN4.N4Weight     = 'all';
optN4.N4Horizon    = 'auto';
optN4.InitialState = 'auto';

%% =========================
% Generate training / held-out dataset
%% =========================
data = call_example_generator(exampleFcn, numExp);

modelName = data.modelName;
Ts        = data.Ts;

[u0Raw, y0Raw] = force_2d_io(data.uCell{1}, data.yCell{1});
ny = size(y0Raw,2);
nu = size(u0Raw,2);

%% =========================
% Build estimation / validation iddata
%% =========================
[zEstMerged, zValCell, yTrueHeldOut] = build_multi_exp_iddata(data, Ts);

fprintf('\n=== Multi-experiment dataset summary ===\n');
fprintf('Number of experiments : %d\n', data.numExp);
fprintf('Input dimension       : %d\n', nu);
fprintf('Output dimension      : %d\n', ny);

%% =========================
% Estimate models of multiple orders
%% =========================
numOrders = numel(orders);
models = cell(numOrders,1);

fitPred_mean  = zeros(numOrders,1);
fitSim_mean   = zeros(numOrders,1);
rmsePred_mean = zeros(numOrders,1);
rmseSim_mean  = zeros(numOrders,1);

fitPred_each  = zeros(numOrders, ny);
fitSim_each   = zeros(numOrders, ny);
rmsePred_each = zeros(numOrders, ny);
rmseSim_each  = zeros(numOrders, ny);

for i = 1:numOrders
    nx = orders(i);
    sys_i = n4sid(zEstMerged, nx, optN4);
    models{i} = sys_i;

    yPredCell = cell(data.numExp,1);
    ySimCell  = cell(data.numExp,1);

    for e = 1:data.numExp
        x0est = findstates(sys_i, zValCell{e}, Inf);

        simOpt  = simOptions('InitialCondition', x0est);
        predOpt = predictOptions('InitialCondition', x0est);

        yPredCell{e} = predict(sys_i, zValCell{e}, 1, predOpt).OutputData;
        ySimCell{e}  = sim(sys_i, zValCell{e}, simOpt).OutputData;
    end

    yPredAll = vertcat(yPredCell{:});
    ySimAll  = vertcat(ySimCell{:});

    metrics_i = compute_prediction_metrics(yTrueHeldOut, yPredAll, ySimAll);

    fitPred_mean(i)  = metrics_i.fit1;
    fitSim_mean(i)   = metrics_i.fitFree;
    rmsePred_mean(i) = metrics_i.rmse1;
    rmseSim_mean(i)  = metrics_i.rmseFree;

    fitPred_each(i,:)  = metrics_i.fit1_each;
    fitSim_each(i,:)   = metrics_i.fitFree_each;
    rmsePred_each(i,:) = metrics_i.rmse1_each;
    rmseSim_each(i,:)  = metrics_i.rmseFree_each;
end

%% =========================
% Pick best order
%% =========================
[~, bestIdx] = max(fitSim_mean);
bestOrder = orders(bestIdx);
sysN4 = models{bestIdx};

%% =========================
% Final held-out validation + plots
%% =========================
resultHeldOut = evaluate_n4sid_dataset_with_plots( ...
    data, 'held-out-validation', 'val', modelName, sysN4, Ts);

%% =========================
% Independent simulation-run test + plots
%% =========================
resultSimRun = [];
if useIndependentSimRunTest
    rng(independentSeed);
    dataSim = call_example_generator(exampleFcn, numSimRunExp);

    resultSimRun = evaluate_n4sid_dataset_with_plots( ...
        dataSim, 'independent-sim-run', independentEvalMode, modelName, sysN4, Ts);
end

%% =========================
% Plot model order selection result
%% =========================
plot_selection_curve( ...
    orders, ...
    fitPred_mean, fitSim_mean, ...
    rmsePred_mean, rmseSim_mean, ...
    'Model order', ...
    [modelName, ' - N4SID order scan']);

%% =========================
% Summary
%% =========================
fprintf('\n=== N4SID summary ===\n');
fprintf('Best order              : %d\n', bestOrder);
fprintf('Held-out 1-step FIT     : %.4f %%\n', resultHeldOut.fit1);
fprintf('Held-out free-run FIT   : %.4f %%\n', resultHeldOut.fitFree);
fprintf('Held-out 1-step RMSE    : %.6f\n', resultHeldOut.rmse1);
fprintf('Held-out free-run RMSE  : %.6f\n', resultHeldOut.rmseFree);

if useIndependentSimRunTest && ~isempty(resultSimRun)
    fprintf('\n=== Independent simulation-run summary ===\n');
    fprintf('Eval mode               : %s\n', independentEvalMode);
    fprintf('1-step FIT              : %.4f %%\n', resultSimRun.fit1);
    fprintf('Free-run FIT            : %.4f %%\n', resultSimRun.fitFree);
    fprintf('1-step RMSE             : %.6f\n', resultSimRun.rmse1);
    fprintf('Free-run RMSE           : %.6f\n', resultSimRun.rmseFree);
end

%% Remove path
rmpath(rootPath);

%% ========================================================================
% Local functions
%% ========================================================================

function data = call_example_generator(exampleFcn, numExp)
% Calls generator robustly whether it accepts only numExp or more inputs.
    try
        data = exampleFcn(numExp);
    catch ME
        error('Failed to call example generator with numExp only: %s', ME.message);
    end
end

function [u, y] = force_2d_io(u, y)
    if isvector(u), u = u(:); end
    if isvector(y), y = y(:); end
end

function [zEstMerged, zValCell, yTrueAll] = build_multi_exp_iddata(data, Ts)

    numExp = data.numExp;
    zEstCell  = cell(numExp,1);
    zValCell  = cell(numExp,1);
    yTrueCell = cell(numExp,1);

    for e = 1:numExp
        [uEst, yEst] = force_2d_io(data.uEstCell{e}, data.yEstCell{e});
        [uVal, yVal] = force_2d_io(data.uValCell{e}, data.yValCell{e});

        zEstCell{e} = iddata(yEst, uEst, Ts);
        zValCell{e} = iddata(yVal, uVal, Ts);
        yTrueCell{e} = yVal;
    end

    zEstMerged = zEstCell{1};
    for e = 2:numExp
        zEstMerged = merge(zEstMerged, zEstCell{e});
    end

    yTrueAll = vertcat(yTrueCell{:});
end

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

    yTrueAll = vertcat(yTrueCell{:});
    yPredAll = vertcat(yPredCell{:});
    ySimAll  = vertcat(ySimCell{:});

    result = compute_prediction_metrics(yTrueAll, yPredAll, ySimAll);
    result.yTrueCell = yTrueCell;
    result.yPredCell = yPredCell;
    result.ySimCell  = ySimCell;
    result.tCell     = tCell;

    plot_pred_sim_per_experiment( ...
        tCell, yTrueCell, yPredCell, ySimCell, ...
        sprintf('%s | %s', modelName, datasetName));

    plot_residual_per_experiment( ...
        tCell, yTrueCell, yPredCell, ySimCell, ...
        sprintf('%s | %s', modelName, datasetName));
end

function plot_pred_sim_per_experiment(tCell, yTrueCell, yPredCell, ySimCell, plotTitlePrefix)

    numExp = numel(tCell);
    ny = size(yTrueCell{1},2);

    for e = 1:numExp
        t = tCell{e};
        yTrue = yTrueCell{e};
        yPred = yPredCell{e};
        ySim  = ySimCell{e};

        figure('Name', sprintf('%s - Exp %d', plotTitlePrefix, e));
        tiledlayout(ny,1);

        for k = 1:ny
            nexttile;
            plot(t, yTrue(:,k), 'k', 'LineWidth', 1.5); hold on;
            plot(t, yPred(:,k), 'b--', 'LineWidth', 1.2);
            plot(t, ySim(:,k),  'r-.', 'LineWidth', 1.2);
            grid on;
            ylabel(sprintf('y_%d', k));

            if k == 1
                title(sprintf('%s | Exp %d', plotTitlePrefix, e));
                legend('True', '1-step Pred', 'Free-run Sim', 'Location', 'best');
            end
        end

        xlabel('Time [s]');
    end
end

function plot_residual_per_experiment(tCell, yTrueCell, yPredCell, ySimCell, plotTitlePrefix)

    numExp = numel(tCell);
    ny = size(yTrueCell{1},2);

    for e = 1:numExp
        t = tCell{e};
        ePred = yTrueCell{e} - yPredCell{e};
        eSim  = yTrueCell{e} - ySimCell{e};

        figure('Name', sprintf('%s Residual - Exp %d', plotTitlePrefix, e));
        tiledlayout(ny,2);

        for k = 1:ny
            nexttile;
            plot(t, ePred(:,k), 'b', 'LineWidth', 1.0);
            grid on;
            title(sprintf('Pred residual y_%d', k));
            ylabel('Error');

            nexttile;
            plot(t, eSim(:,k), 'r', 'LineWidth', 1.0);
            grid on;
            title(sprintf('Sim residual y_%d', k));
        end

        sgtitle(sprintf('%s | Exp %d Residual', plotTitlePrefix, e));
    end
end

function metrics = compute_prediction_metrics(yTrue, yPred, ySim)

    yTrue = double(yTrue);
    yPred = double(yPred);
    ySim  = double(ySim);

    [N, ny] = size(yTrue); %#ok<ASGLU>

    rmse1_each    = sqrt(mean((yTrue - yPred).^2, 1));
    rmseFree_each = sqrt(mean((yTrue - ySim ).^2, 1));

    fit1_each    = zeros(1, ny);
    fitFree_each = zeros(1, ny);

    for k = 1:ny
        denom = norm(yTrue(:,k) - mean(yTrue(:,k)));
        if denom < eps
            fit1_each(k)    = NaN;
            fitFree_each(k) = NaN;
        else
            fit1_each(k)    = 100 * (1 - norm(yTrue(:,k) - yPred(:,k)) / denom);
            fitFree_each(k) = 100 * (1 - norm(yTrue(:,k) - ySim(:,k))  / denom);
        end
    end

    metrics.rmse1_each    = rmse1_each;
    metrics.rmseFree_each = rmseFree_each;
    metrics.fit1_each     = fit1_each;
    metrics.fitFree_each  = fitFree_each;

    metrics.rmse1    = mean(rmse1_each, 'omitnan');
    metrics.rmseFree = mean(rmseFree_each, 'omitnan');
    metrics.fit1     = mean(fit1_each, 'omitnan');
    metrics.fitFree  = mean(fitFree_each, 'omitnan');
end

function plot_selection_curve(orders, fitPred, fitSim, rmsePred, rmseSim, xLabelStr, figTitleStr)

    figure('Name', figTitleStr);
    tiledlayout(2,1);

    nexttile;
    plot(orders, fitPred, 'o--', 'LineWidth', 1.2); hold on;
    plot(orders, fitSim,  's-.', 'LineWidth', 1.2);
    grid on;
    xlabel(xLabelStr);
    ylabel('FIT [%]');
    title(figTitleStr);
    legend('1-step prediction', 'free-run simulation', 'Location', 'best');

    nexttile;
    plot(orders, rmsePred, 'o--', 'LineWidth', 1.2); hold on;
    plot(orders, rmseSim,  's-.', 'LineWidth', 1.2);
    grid on;
    xlabel(xLabelStr);
    ylabel('RMSE');
    legend('1-step prediction', 'free-run simulation', 'Location', 'best');
end