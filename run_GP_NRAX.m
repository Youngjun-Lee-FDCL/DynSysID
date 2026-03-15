clear; clc; close all;

%% Add GPML path
addpath(genpath('./'));
startup;

rng(1);

%% =========================
% Select example generator
%% =========================
% exampleFcn = @generate_frigola_benchmark_example;
exampleFcn = @generate_linear_msd_example;
% exampleFcn = @generate_toy_nonlinear_example;
% exampleFcn = @generate_nonlinear_msd_example;
% exampleFcn = @generate_nonlinear_twotank_example;
% exampleFcn = @generate_linear_mimo3_example;
% exampleFcn = @generate_nonlinear_cstr_PID_example;
% exampleFcn = @generate_nonlinear_aircraft43_PID_example;

%% =========================
% Generate raw data
%% =========================
data = exampleFcn();

t         = data.t;
u         = data.u;     % N x nu
y         = data.y;     % N x ny
na_input  = data.na;
nb_input  = data.nb;
nk_input  = data.nk;
Ts        = data.Ts;
modelName = data.modelName;

if isvector(u), u = u(:); end
if isvector(y), y = y(:); end

N  = size(y,1);
ny = size(y,2);
nu = size(u,2);

%% =========================
% Convert orders to full matrices
%% =========================
[naMat, nbMat, nkMat] = expand_narx_orders(na_input, nb_input, nk_input, ny, nu);

%% =========================
% Build universal NARX regressors
%% =========================
[Xall, Yall, idxAll] = build_mimo_narx_regressors(y, u, naMat, nbMat, nkMat);

%% =========================
% Train/test split
%% =========================
if isfield(data, 'uEst') && isfield(data, 'yEst')
    NtrOriginal = size(data.uEst,1);
    trainMask = idxAll <= NtrOriginal;
    testMask  = idxAll >  NtrOriginal;
elseif isfield(data, 'idxTe')
    testMask  = ismember(idxAll, data.idxTe(:));
    trainMask = ~testMask;
elseif isfield(data, 'idxVal')
    testMask  = ismember(idxAll, data.idxVal(:));
    trainMask = ~testMask;
else
    Ntr = round(0.7 * size(Xall,1));
    trainMask = false(size(Xall,1),1);
    trainMask(1:Ntr) = true;
    testMask = ~trainMask;
end

Xtr = Xall(trainMask,:);
Ytr = Yall(trainMask,:);
Xte = Xall(testMask,:);
Yte = Yall(testMask,:);
idxTe = idxAll(testMask);

%% =========================
% Normalize
%% =========================
[XtrN, muX, stdX] = normalize_data(Xtr);
XteN = apply_normalization(Xte, muX, stdX);

[YtrN, muY, stdY] = normalize_data(Ytr);
YteN = apply_normalization(Yte, muY, stdY); %#ok<NASGU>

%% =========================
% GPML setup
%% =========================
meanfunc = @meanZero;
covfunc  = @covSEard;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

D = size(XtrN,2);

%% =========================
% Train one GP per output channel
%% =========================
hypOpt = cell(ny,1);
muTeN  = zeros(size(Xte,1), ny);
s2TeN  = zeros(size(Xte,1), ny);

for j = 1:ny
    hyp = struct();
    hyp.mean = [];
    hyp.cov  = [zeros(D,1); 0];
    hyp.lik  = log(0.1);

    hypOpt{j} = minimize(hyp, @gp, -120, ...
        inffunc, meanfunc, covfunc, likfunc, XtrN, YtrN(:,j));

    [muTeN(:,j), s2TeN(:,j)] = gp(hypOpt{j}, ...
        inffunc, meanfunc, covfunc, likfunc, XtrN, YtrN(:,j), XteN);
end

muTe = muTeN .* stdY + muY;
s2Te = s2TeN .* (stdY.^2);

%% =========================
% Free-run simulation
%% =========================
yHatFree = nan(size(Yte));     % Ntest x ny
testStartOriginalIndex = idxTe(1);

for n = 1:size(Yte,1)
    kOrig = idxTe(n);

    xNow = build_single_mimo_regressor_for_rollout( ...
        kOrig, testStartOriginalIndex, y, u, yHatFree, ...
        naMat, nbMat, nkMat);

    xNowN = apply_normalization(xNow, muX, stdX);

    for j = 1:ny
        muNowN = gp(hypOpt{j}, inffunc, meanfunc, covfunc, likfunc, ...
            XtrN, YtrN(:,j), xNowN);
        yHatFree(n,j) = muNowN * stdY(j) + muY(j);
    end
end

%% =========================
% Common metrics
%% =========================
metrics = compute_prediction_metrics(Yte, muTe, yHatFree);

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

%% =========================
% Plot measured outputs (common)
%% =========================
plot_measured_outputs(t, y, modelName);

%% =========================
% Plot one-step with uncertainty band (GP-specific)
%% =========================
plot_gp_prediction_with_band( ...
    tTe, Yte, muTe, s2Te, ...
    rmse1_each, fit1_each, ...
    modelName, 'GP-NARX', 'One-step prediction');

%% =========================
% Plot free-run simulation (common)
%% =========================
plot_estimation_result( ...
    tTe, Yte, yHatFree, ...
    rmseFree_each, fitFree_each, ...
    modelName, 'GP-NARX', ...
    'Free-run simulation', ...
    'free-run', ...
    'b--');

%% =========================
% Plot errors (common)
%% =========================
plot_estimation_errors(tTe, Yte, muTe, yHatFree, modelName, 'GP-NARX');

%% =========================
% Print summary (common)
%% =========================
algoName = 'GP-NARX';
extraInfo = sprintf('Regressor dimension : %d', size(Xtr,2));

print_sysid_summary( ...
    modelName, algoName, nu, ny, ...
    rmse1, fit1, rmseFree, fitFree, ...
    rmse1_each, fit1_each, rmseFree_each, fitFree_each, ...
    extraInfo);

%% remove path
rmpath(genpath('./'));