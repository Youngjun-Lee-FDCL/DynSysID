clear; clc; close all;

%% Add GPML path
addpath(genpath('./'));
startup;
rng(1);

%% =========================
% Select example generator
%% =========================
% exampleFcn = @generate_frigola_benchmark_example;
% exampleFcn = @generate_linear_msd_example;
% exampleFcn = @generate_toy_nonlinear_example;
% exampleFcn = @generate_nonlinear_msd_example;
% exampleFcn = @generate_nonlinear_twotank_example;
% exampleFcn = @generate_linear_mimo3_example;
% exampleFcn = @generate_nonlinear_cstr_PID_example;
exampleFcn = @generate_nonlinear_aircraft43_PID_example;
% exampleFcn = @generate_vgpssm_easy_example;

%% =========================
% Multi-experiment options
%% =========================
numExp = 5;              % number of independent experiments
valExpForTest = numExp;  % which experiment to use for final validation/free-run

%% =========================
% Generate raw data
%% =========================
data = exampleFcn(numExp);

na_input  = data.na;
nb_input  = data.nb;
nk_input  = data.nk;
modelName = data.modelName;

%% =========================
% Basic dimensions from first experiment
%% =========================
t0    = data.tCell{1};
u0Raw = data.uCell{1};
y0Raw = data.yCell{1};

if isvector(u0Raw), u0Raw = u0Raw(:); end
if isvector(y0Raw), y0Raw = y0Raw(:); end

ny = size(y0Raw,2);
nu = size(u0Raw,2);

%% =========================
% Output-only preprocessing
%% =========================
usePreprocessing = false;
sgolayFrame_y    = 21;   % Must be odd

%% =========================
% Convert orders to full matrices
%% =========================
[naMat, nbMat, nkMat] = expand_narx_orders(na_input, nb_input, nk_input, ny, nu);

%% =========================
% Optional train-data sampling using BlockCoverageSampler
% Sampling is performed per experiment on regressor sequence
%% =========================
useBlockCoverageSampling = true;
blockLenSampling         = 30;
numBlocksKeepSampling    = 3;
useOverlapSampling       = true;

%% =========================
% Build multi-experiment training regressors
%% =========================
Xtr = [];
Ytr = [];

numTrainRowsBeforeSampling = 0;
numTrainRowsAfterSampling  = 0;

idxKeepPerExp = cell(data.numExp,1);
idxAllPerExp  = cell(data.numExp,1);

for e = 1:data.numExp
    % Use estimation part only for training
    uEstRaw = data.uEstCell{e};
    yEstRaw = data.yEstCell{e};

    if isvector(uEstRaw), uEstRaw = uEstRaw(:); end
    if isvector(yEstRaw), yEstRaw = yEstRaw(:); end

    Ne = size(yEstRaw,1);

    % Preprocess output only
    ue = uEstRaw;
    ye = yEstRaw;

    if usePreprocessing
        frameNow = min(sgolayFrame_y, Ne);
        if mod(frameNow,2) == 0
            frameNow = frameNow - 1;
        end
        if frameNow < 3
            error('Savitzky-Golay frame length for output is too short.');
        end

        for j = 1:ny
            ye(:,j) = smoothdata(yEstRaw(:,j), 'sgolay', frameNow);
        end
    end

    % Build experiment-wise regressors
    [Xe, Ye, idxAll_e] = build_mimo_narx_regressors(ye, ue, naMat, nbMat, nkMat);
    
    idxAllPerExp{e} = idxAll_e;

    numTrainRowsBeforeSampling = numTrainRowsBeforeSampling + size(Xe,1);

    % Optional experiment-wise block coverage sampling
    if useBlockCoverageSampling
        sampler = BlockCoverageSampler( ...
            blockLenSampling, ...
            numBlocksKeepSampling, ...
            useOverlapSampling);

        [idxKeepTrain, sampler] = sampler.selectBlocksFromRegressor(Xe, Ye);
        
        idxKeepPerExp{e} = idxKeepTrain;

        Xe = Xe(idxKeepTrain,:);
        Ye = Ye(idxKeepTrain,:);
    else
        idxKeepPerExp{e} = (1:size(Xe,1)).';
    end

    numTrainRowsAfterSampling = numTrainRowsAfterSampling + size(Xe,1);

    % Concatenate after experiment-wise processing
    Xtr = [Xtr; Xe];
    Ytr = [Ytr; Ye];
end
plot_selected_experiment_regions_io(data, idxKeepPerExp, idxAllPerExp);

fprintf('\n=== Multi-experiment training set summary ===\n');
fprintf('Number of experiments used for training : %d\n', data.numExp);
fprintf('Training rows before sampling           : %d\n', numTrainRowsBeforeSampling);
fprintf('Training rows after sampling            : %d\n', numTrainRowsAfterSampling);
fprintf('Keep ratio                              : %.2f %%\n', ...
    100 * numTrainRowsAfterSampling / numTrainRowsBeforeSampling);

%% =========================
% Build test regressors from one selected validation experiment
%% =========================
if valExpForTest < 1 || valExpForTest > data.numExp
    error('valExpForTest is out of range.');
end

t         = data.tCell{valExpForTest};
uRaw      = data.uCell{valExpForTest};
yRaw      = data.yCell{valExpForTest};

uValRaw   = data.uValCell{valExpForTest};
yValRaw   = data.yValCell{valExpForTest};
idxValRaw = data.idxValCell{valExpForTest};

if isvector(uRaw),    uRaw = uRaw(:); end
if isvector(yRaw),    yRaw = yRaw(:); end
if isvector(uValRaw), uValRaw = uValRaw(:); end
if isvector(yValRaw), yValRaw = yValRaw(:); end

N = size(yRaw,1);

% Preprocess full sequence for rollout/history usage
u = uRaw;
y = yRaw;

if usePreprocessing
    frameNow = min(sgolayFrame_y, N);
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

% Build regressors from full selected experiment
[XallTestExp, YallTestExp, idxAllTestExp] = build_mimo_narx_regressors(y, u, naMat, nbMat, nkMat);

% Select validation/test rows only
testMask = ismember(idxAllTestExp, idxValRaw);

Xte   = XallTestExp(testMask,:);
Yte   = YallTestExp(testMask,:);   % Possibly preprocessed target
idxTe = idxAllTestExp(testMask);

if isempty(idxTe)
    error('Test set is empty.');
end

% Raw targets for final evaluation and plotting
YteRaw = yRaw(idxTe,:);

%% =========================
% Normalize
%% =========================
[XtrN, muX, stdX] = normalize_data(Xtr);
XteN = apply_normalization(Xte, muX, stdX);

[YtrN, muY, stdY] = normalize_data(Ytr);

%% =========================
% GPML setup
%% =========================
meanfunc = {@meanZero};
covfunc  = {@covSEard};
likfunc  = {@likGauss};
inffunc  = @infGaussLik;

D = size(XtrN,2);

%% =========================
% Train one GP per output channel
% Also precompute posterior quantities for fast prediction
%% =========================
hypOpt   = cell(ny,1);
postCell = cell(ny,1);

muTeN = zeros(size(XteN,1), ny);
s2TeN = zeros(size(XteN,1), ny);

for j = 1:ny
    hyp = struct();
    hyp.mean = [];
    hyp.cov  = [zeros(D,1); 0];
    hyp.lik  = log(0.1);

    % Hyperparameter optimization
    hypOpt{j} = minimize(hyp, @gp, -200, ...
        inffunc, meanfunc, covfunc, likfunc, XtrN, YtrN(:,j));

    % Posterior for fast prediction
    [postCell{j}, ~, ~] = infGaussLik( ...
        hypOpt{j}, meanfunc, covfunc, likfunc, XtrN, YtrN(:,j));

    % One-step prediction on test regressors
    [muTeN(:,j), s2TeN(:,j)] = gpml_predict_mean_var_exact( ...
        hypOpt{j}, meanfunc, covfunc, XtrN, XteN, postCell{j});
end

muTe = muTeN .* stdY + muY;
s2Te = s2TeN .* (stdY.^2);

%% =========================
% Free-run simulation
% Rollout uses selected validation experiment only
%% =========================
disp("Free-run simulation starts...")
tFreeRunStart = tic;

yHatFree = nan(size(Yte));   % Predicted output on original output scale
testStartOriginalIndex = idxTe(1);

for n = 1:size(Yte,1)
    kOrig = idxTe(n);

    xNow = build_single_mimo_regressor_for_rollout( ...
        kOrig, testStartOriginalIndex, y, u, yHatFree, ...
        naMat, nbMat, nkMat);

    xNowN = apply_normalization(xNow, muX, stdX);

    for j = 1:ny
        muNowN = gpml_predict_mean_exact_single( ...
            hypOpt{j}, meanfunc, covfunc, XtrN, xNowN, postCell{j});

        yHatFree(n,j) = muNowN * stdY(j) + muY(j);
    end
end

tFreeRunElapsed = toc(tFreeRunStart);
fprintf('Free-run simulation runtime: %.6f seconds\n', tFreeRunElapsed);

%% =========================
% Metrics against RAW outputs
%% =========================
metrics = compute_prediction_metrics(YteRaw, muTe, yHatFree);

rmse1_each    = metrics.rmse1_each;
fit1_each     = metrics.fit1_each;
rmseFree_each = metrics.rmseFree_each;
fitFree_each  = metrics.fitFree_each;

rmse1    = metrics.rmse1;
fit1     = metrics.fit1;
rmseFree = metrics.rmseFree;
fitFree  = metrics.fitFree;

%% =========================
% Validation time
%% =========================
tTe = t(idxTe);

if usePreprocessing
    algoName = sprintf('GP-NARX multi-exp (%d exp, test exp %d, y-only Savitzky-Golay)', ...
        data.numExp, valExpForTest);
else
    algoName = sprintf('GP-NARX multi-exp (%d exp, test exp %d)', ...
        data.numExp, valExpForTest);
end

%% =========================
% Plot measured outputs
%% =========================
if usePreprocessing
    plot_measured_outputs_with_preprocessed(t, yRaw, y, modelName);
else
    plot_measured_outputs(t, yRaw, modelName);
end

%% =========================
% Plot one-step prediction with raw target
%% =========================
plot_gp_prediction_with_band( ...
    tTe, YteRaw, muTe, s2Te, ...
    rmse1_each, fit1_each, ...
    modelName, algoName, 'One-step prediction (evaluated on raw output)');

%% =========================
% Plot free-run simulation with raw target
%% =========================
plot_estimation_result( ...
    tTe, YteRaw, yHatFree, ...
    rmseFree_each, fitFree_each, ...
    modelName, algoName, ...
    'Free-run simulation (evaluated on raw output)', ...
    'free-run', ...
    'b--');

%% =========================
% Plot errors against raw target
%% =========================
plot_estimation_errors(tTe, YteRaw, muTe, yHatFree, modelName, algoName);

%% =========================
% Print summary
%% =========================
if useBlockCoverageSampling
    samplingFlagStr = 'enabled';
else
    samplingFlagStr = 'disabled';
end

if usePreprocessing
    extraInfo = sprintf(['Regressor dimension : %d\n' ...
                         'Number of experiments: %d\n' ...
                         'Test experiment idx : %d\n' ...
                         'Input preprocessing : none\n' ...
                         'Output preprocessing: Savitzky-Golay\n' ...
                         'SG frame (output)   : %d\n' ...
                         'Prediction mode     : cached posterior mean/var\n' ...
                         'Train sampling      : %s\n' ...
                         'Sampling block len  : %d\n' ...
                         'Sampling #blocks    : %d\n' ...
                         'Train rows before   : %d\n' ...
                         'Train rows after    : %d\n' ...
                         'Metrics target      : raw output'], ...
                         size(Xtr,2), data.numExp, valExpForTest, ...
                         sgolayFrame_y, ...
                         samplingFlagStr, ...
                         blockLenSampling, numBlocksKeepSampling, ...
                         numTrainRowsBeforeSampling, numTrainRowsAfterSampling);
else
    extraInfo = sprintf(['Regressor dimension : %d\n' ...
                         'Number of experiments: %d\n' ...
                         'Test experiment idx : %d\n' ...
                         'Input preprocessing : none\n' ...
                         'Output preprocessing: none\n' ...
                         'Prediction mode     : cached posterior mean/var\n' ...
                         'Train sampling      : %s\n' ...
                         'Sampling block len  : %d\n' ...
                         'Sampling #blocks    : %d\n' ...
                         'Train rows before   : %d\n' ...
                         'Train rows after    : %d\n' ...
                         'Metrics target      : raw output'], ...
                         size(Xtr,2), data.numExp, valExpForTest, ...
                         samplingFlagStr, ...
                         blockLenSampling, numBlocksKeepSampling, ...
                         numTrainRowsBeforeSampling, numTrainRowsAfterSampling);
end

print_sysid_summary( ...
    modelName, algoName, nu, ny, ...
    rmse1, fit1, rmseFree, fitFree, ...
    rmse1_each, fit1_each, rmseFree_each, fitFree_each, ...
    extraInfo);

%% Remove path
rmpath(genpath('./'));