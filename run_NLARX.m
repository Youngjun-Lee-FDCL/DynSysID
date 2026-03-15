clear; clc; close all;

rng(1);

addpath(genpath('./'));

%% =========================
% Select example generator
%% =========================
% exampleFcn = @generate_toy_nonlinear_example;
% exampleFcn = @generate_linear_msd_example;
exampleFcn = @generate_nonlinear_msd_example;
% exampleFcn = @generate_frigola_benchmark_example;
% exampleFcn = @generate_linear_mimo3_example;
% exampleFcn = @generate_linear_mimo3_PID_example;
% exampleFcn = @generate_nonlinear_twotank_example;

%% =========================
% Generate data
%% =========================
data = exampleFcn();

t         = data.t;
u         = data.u;
y         = data.y;
Ts        = data.Ts;
na        = data.na;
nb        = data.nb;
nk        = data.nk;
modelName = data.modelName;

% force 2D shape
if isvector(u), u = u(:); end
if isvector(y), y = y(:); end

N  = size(y,1);
ny = size(y,2);
nu = size(u,2);

%% =========================
% Build iddata objects
%% =========================
z = iddata(y, u, Ts);

% Split rule:
% 1) if example provides estimation/validation split explicitly, use it
% 2) else if idxTe or idxVal exists, use that
% 3) else use 70/30 split
if isfield(data, 'uEst') && isfield(data, 'yEst') && isfield(data, 'uVal') && isfield(data, 'yVal')
    zEst = iddata(data.yEst, data.uEst, Ts);
    zVal = iddata(data.yVal, data.uVal, Ts);

    if isfield(data, 'idxVal')
        tVal = data.t(data.idxVal);
    else
        tVal = t(size(data.uEst,1)+1:end);
    end

elseif isfield(data, 'idxVal')
    idxVal = data.idxVal(:);
    NtrOriginal = idxVal(1) - 1;
    zEst = z(1:NtrOriginal);
    zVal = z(NtrOriginal+1:end);
    tVal = t(NtrOriginal+1:end);

elseif isfield(data, 'idxTe')
    idxTe = data.idxTe(:);
    NtrOriginal = idxTe(1) - 1;
    zEst = z(1:NtrOriginal);
    zVal = z(NtrOriginal+1:end);
    tVal = t(NtrOriginal+1:end);

else
    NtrOriginal = round(0.7 * N);
    zEst = z(1:NtrOriginal);
    zVal = z(NtrOriginal+1:end);
    tVal = t(NtrOriginal+1:end);
end

%% =========================
% Choose nonlinear output function
%% =========================
% Available choices include:
%   idWaveletNetwork
%   idTreePartition
%   idSigmoidNetwork
%   idNeuralNetwork   (newer releases)
outputFcn = idWaveletNetwork;

% Alternatives:
% outputFcn = idTreePartition;
% outputFcn = idSigmoidNetwork;

%% =========================
% Estimation options
%% =========================
optNL = nlarxOptions;
optNL.SearchOptions.MaxIterations = 50;
optNL.Focus = 'simulation';

%% =========================
% Estimate NLARX model
%% =========================
% For MIMO:
%   na : ny-by-ny
%   nb : ny-by-nu
%   nk : ny-by-nu
% For SISO:
%   scalars are also fine
sysNLARX = nlarx(zEst, [na nb nk], outputFcn, optNL);

%% =========================
% One-step prediction on validation set
%% =========================
yPredObj = predict(sysNLARX, zVal, 1);
yPred = yPredObj.OutputData;   % Nval x ny

yTrue = zVal.OutputData;       % Nval x ny

%% =========================
% Free-run simulation on validation set
%% =========================
x0 = findstates(sysNLARX, zVal);
optSim = simOptions;
optSim.InitialCondition = x0;

ySimObj = sim(sysNLARX, zVal, optSim);
ySim = ySimObj.OutputData;     % Nval x ny

%% =========================
% Common metrics
%% =========================
metrics = compute_prediction_metrics(yTrue, yPred, ySim);

rmse1_each    = metrics.rmse1_each;
rmseFree_each = metrics.rmseFree_each;
fit1_each     = metrics.fit1_each;
fitFree_each  = metrics.fitFree_each;

rmse1    = metrics.rmse1;
rmseFree = metrics.rmseFree;
fit1     = metrics.fit1;
fitFree  = metrics.fitFree;

%% =========================
% Plot measured outputs (common)
%% =========================
plot_measured_outputs(t, y, modelName);

%% =========================
% Plot one-step prediction (common)
%% =========================
plot_estimation_result( ...
    tVal, yTrue, yPred, ...
    rmse1_each, fit1_each, ...
    modelName, 'NLARX', ...
    'One-step prediction', ...
    sprintf('one-step (%s)', class(outputFcn)), ...
    'r--');

%% =========================
% Plot free-run simulation (common)
%% =========================
plot_estimation_result( ...
    tVal, yTrue, ySim, ...
    rmseFree_each, fitFree_each, ...
    modelName, 'NLARX', ...
    'Free-run simulation', ...
    sprintf('free-run (%s)', class(outputFcn)), ...
    'b--');

%% =========================
% Plot errors (common)
%% =========================
plot_estimation_errors(tVal, yTrue, yPred, ySim, modelName, 'NLARX');

%% =========================
% Optional: model info
%% =========================
disp(sysNLARX);

%% =========================
% Print summary (common)
%% =========================
algoName = 'NLARX';
extraInfo = sprintf('NLARX type         : %s', class(outputFcn));

print_sysid_summary( ...
    modelName, algoName, nu, ny, ...
    rmse1, fit1, rmseFree, fitFree, ...
    rmse1_each, fit1_each, rmseFree_each, fitFree_each, ...
    extraInfo);

%% Remove path
rmpath(genpath('./'));