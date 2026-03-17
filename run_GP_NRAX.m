clear; clc; close all;

%% Add GPML path
addpath(genpath('./'));
startup;
rng(1);

%% =========================
% User-editable settings
% (Keep performance-affecting variables HERE)
%% =========================

% Example generator
% exampleFcn = @generate_nonlinear_msd_example;
exampleFcn = @generate_frigola_benchmark_example;
% exampleFcn = @generate_linear_msd_example;
% exampleFcn = @generate_toy_nonlinear_example;
% exampleFcn = @generate_nonlinear_twotank_example;
% exampleFcn = @generate_linear_mimo3_example;
% exampleFcn = @generate_nonlinear_cstr_PID_example;
% exampleFcn = @generate_nonlinear_aircraft43_PID_example;
% exampleFcn = @generate_vgpssm_easy_example;

% Dataset sizes
numExp       = 5;    % training / held-out validation experiments
numSimRunExp = 3;    % independent simulation-run experiments

% Independent test option
useIndependentSimRunTest = true;
independentEvalMode      = 'full';   % 'full' or 'val'
independentSeed          = 100;      % meaningful only if exampleFcn does not fix rng internally

% Preprocessing
usePreprocessing = false;
sgolayFrame_y    = 21;

% Sampling
useBlockCoverageSampling = true;
blockLenSampling         = 30;
numBlocksKeepSampling    = 5;
useOverlapSampling       = true;

% GP setup
meanfunc = {@meanZero};
covfunc  = {@covSEard};
likfunc  = {@likGauss};
inffunc  = @infGaussLik;
numMinimizeIter = -200;   % GP hyperparameter optimization iterations

%% =========================
% Generate training / held-out dataset
%% =========================
data = exampleFcn(numExp);

na_input  = data.na;
nb_input  = data.nb;
nk_input  = data.nk;
modelName = data.modelName;

[u0Raw, y0Raw] = force_2d_io(data.uCell{1}, data.yCell{1});
ny = size(y0Raw,2);
nu = size(u0Raw,2);

[naMat, nbMat, nkMat] = expand_narx_orders(na_input, nb_input, nk_input, ny, nu);

%% =========================
% Build training set
%% =========================
trainOpt = struct( ...
    'usePreprocessing', usePreprocessing, ...
    'sgolayFrame_y', sgolayFrame_y, ...
    'useBlockCoverageSampling', useBlockCoverageSampling, ...
    'blockLenSampling', blockLenSampling, ...
    'numBlocksKeepSampling', numBlocksKeepSampling, ...
    'useOverlapSampling', useOverlapSampling);

trainSet = build_gp_narx_training_set( ...
    data, naMat, nbMat, nkMat, trainOpt);

plot_selected_experiment_regions_io(data, trainSet.idxKeepPerExp, trainSet.idxAllPerExp);

fprintf('\n=== Multi-experiment training set summary ===\n');
fprintf('Number of experiments used for training : %d\n', data.numExp);
fprintf('Training rows before sampling           : %d\n', trainSet.numRowsBeforeSampling);
fprintf('Training rows after sampling            : %d\n', trainSet.numRowsAfterSampling);
fprintf('Keep ratio                              : %.2f %%\n', ...
    100 * trainSet.numRowsAfterSampling / trainSet.numRowsBeforeSampling);

%% =========================
% Normalize
%% =========================
[XtrN, muX, stdX] = normalize_data(trainSet.Xtr);
[YtrN, muY, stdY] = normalize_data(trainSet.Ytr);

%% =========================
% Train GP models
%% =========================
gpOpt = struct( ...
    'meanfunc', meanfunc, ...
    'covfunc', covfunc, ...
    'likfunc', likfunc, ...
    'inffunc', inffunc, ...
    'numMinimizeIter', numMinimizeIter);

gpModel = train_gp_narx_models(XtrN, YtrN, gpOpt);

%% =========================
% Held-out validation on original dataset
%% =========================
evalOpt = struct( ...
    'usePreprocessing', usePreprocessing, ...
    'sgolayFrame_y', sgolayFrame_y, ...
    'meanfunc', meanfunc, ...
    'covfunc', covfunc);

normStat = struct( ...
    'XtrN', XtrN, ...
    'muX', muX, 'stdX', stdX, ...
    'muY', muY, 'stdY', stdY);

resultHeldOut = evaluate_gp_narx_dataset( ...
    data, 'heldout-val', 'val', modelName, ...
    naMat, nbMat, nkMat, ...
    gpModel, normStat, evalOpt);

%% =========================
% Independent simulation-run test
%% =========================
if useIndependentSimRunTest
    rng(independentSeed);
    dataSim = exampleFcn(numSimRunExp);

    resultSimRun = evaluate_gp_narx_dataset( ...
        dataSim, 'independent-sim-run', independentEvalMode, modelName, ...
        naMat, nbMat, nkMat, ...
        gpModel, normStat, evalOpt);
end

%% =========================
% Summary
%% =========================
print_run_summary( ...
    modelName, data.numExp, numSimRunExp, ...
    trainSet, resultHeldOut, resultSimRun, ...
    useIndependentSimRunTest, independentEvalMode, ...
    usePreprocessing, useBlockCoverageSampling, ...
    blockLenSampling, numBlocksKeepSampling, ...
    nu, ny);

%% Remove path
rmpath(genpath('./'));