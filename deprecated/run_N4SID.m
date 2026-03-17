clear; clc; close all;

rng(1);

addpath(genpath('./'));

%% =========================
% Select example generator
%% =========================
% exampleFcn = @generate_linear_msd_example;
% exampleFcn = @generate_nonlinear_msd_example;
% exampleFcn = @generate_toy_nonlinear_example;
% exampleFcn = @generate_frigola_benchmark_example;
% exampleFcn = @generate_linear_mimo3_example;
% exampleFcn = @generate_linear_mimo3_PID_example;
% exampleFcn = @generate_nonlinear_twotank_example;
% exampleFcn = @generate_nonlinear_aircraft43_PID_example;
exampleFcn = @generate_vgpssm_easy_example;
%% =========================
% Generate data
%% =========================
data = exampleFcn();

t         = data.t;
u         = data.u;
y         = data.y;
Ts        = data.Ts;
modelName = data.modelName;

% force 2D
if isvector(u), u = u(:); end
if isvector(y), y = y(:); end

N  = size(y,1);
nu = size(u,2);
ny = size(y,2);

%% =========================
% Build iddata objects
%% =========================
z = iddata(y, u, Ts);

% Split rule:
% 1) explicit estimation/validation split
% 2) idxVal
% 3) idxTe
% 4) fallback 70/30
if isfield(data, 'uEst') && isfield(data, 'yEst') && isfield(data, 'uVal') && isfield(data, 'yVal')
    zEst = iddata(data.yEst, data.uEst, Ts);
    zVal = iddata(data.yVal, data.uVal, Ts);
    if isfield(data, 'idxVal')
        tVal = t(data.idxVal);
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

yTrue = zVal.OutputData;   % Nval x ny

%% =========================
% Choose model order
%% =========================
orders = 1:8;

%% =========================
% N4SID options
%% =========================
optN4 = n4sidOptions;
optN4.Focus = 'simulation';
optN4.N4Weight = 'all';      % alternatives: 'MOESP', 'SSARX', 'auto'
optN4.N4Horizon = 'auto';
optN4.InitialState = 'auto';

%% =========================
% Estimate models of multiple orders
%% =========================
models = cell(numel(orders),1);

fitPred_each   = zeros(numel(orders), ny);
fitSim_each    = zeros(numel(orders), ny);
rmsePred_each  = zeros(numel(orders), ny);
rmseSim_each   = zeros(numel(orders), ny);

fitPred_mean   = zeros(numel(orders),1);
fitSim_mean    = zeros(numel(orders),1);
rmsePred_mean  = zeros(numel(orders),1);
rmseSim_mean   = zeros(numel(orders),1);

for i = 1:numel(orders)
    nx = orders(i);

    models{i} = n4sid(zEst, nx, optN4);

    % one-step prediction
    yPredObj = predict(models{i}, zVal, 1);
    yPred_i = yPredObj.OutputData;

    % free-run simulation
    ySimObj = sim(models{i}, zVal);
    ySim_i = ySimObj.OutputData;

    % common metric function
    metrics_i = compute_prediction_metrics(yTrue, yPred_i, ySim_i);

    rmsePred_each(i,:) = metrics_i.rmse1_each;
    rmseSim_each(i,:)  = metrics_i.rmseFree_each;
    fitPred_each(i,:)  = metrics_i.fit1_each;
    fitSim_each(i,:)   = metrics_i.fitFree_each;

    fitPred_mean(i)   = metrics_i.fit1;
    fitSim_mean(i)    = metrics_i.fitFree;
    rmsePred_mean(i)  = metrics_i.rmse1;
    rmseSim_mean(i)   = metrics_i.rmseFree;
end

%% =========================
% Pick best order based on mean free-run FIT
%% =========================
[~, bestIdx] = max(fitSim_mean);
bestOrder = orders(bestIdx);
sysN4 = models{bestIdx};

%% =========================
% Final one-step / free-run using selected model
%% =========================
x0 = findstates(sysN4, zVal);
optSim = simOptions('InitialCondition', x0);

yPredObj = predict(sysN4, zVal, 1, optSim);
yPred = yPredObj.OutputData;

ySimObj = sim(sysN4, zVal, optSim);
ySim = ySimObj.OutputData;

metrics_best = compute_prediction_metrics(yTrue, yPred, ySim);

rmsePred_best_each = metrics_best.rmse1_each;
rmseSim_best_each  = metrics_best.rmseFree_each;
fitPred_best_each  = metrics_best.fit1_each;
fitSim_best_each   = metrics_best.fitFree_each;

rmsePred_best = metrics_best.rmse1;
rmseSim_best  = metrics_best.rmseFree;
fitPred_best  = metrics_best.fit1;
fitSim_best   = metrics_best.fitFree;

%% =========================
% Plot measured output (common)
%% =========================
plot_measured_outputs(t, y, modelName);

%% =========================
% Plot order selection results (common)
%% =========================
plot_selection_curve( ...
    orders, ...
    fitPred_mean, fitSim_mean, ...
    rmsePred_mean, rmseSim_mean, ...
    'Model order', ...
    [modelName, ' - N4SID order scan']);

%% =========================
% Plot one-step prediction (common)
%% =========================
plot_estimation_result( ...
    tVal, yTrue, yPred, ...
    rmsePred_best_each, fitPred_best_each, ...
    modelName, 'N4SID', ...
    'One-step prediction', ...
    sprintf('One-step | order=%d', bestOrder), ...
    'r--');

%% =========================
% Plot free-run simulation (common)
%% =========================
plot_estimation_result( ...
    tVal, yTrue, ySim, ...
    rmseSim_best_each, fitSim_best_each, ...
    modelName, 'N4SID', ...
    'Free-run simulation', ...
    sprintf('Free-run | order=%d', bestOrder), ...
    'b--');

%% =========================
% Plot errors (common)
%% =========================
plot_estimation_errors(tVal, yTrue, yPred, ySim, modelName, 'N4SID');

%% =========================
% Optional: model info
%% =========================
disp(sysN4);

%% =========================
% Print summary (common)
%% =========================
algoName = 'N4SID';
extraInfo = sprintf('Best order         : %d', bestOrder);

print_sysid_summary( ...
    modelName, algoName, nu, ny, ...
    rmsePred_best, fitPred_best, rmseSim_best, fitSim_best, ...
    rmsePred_best_each, fitPred_best_each, rmseSim_best_each, fitSim_best_each, ...
    extraInfo);

%% Remove path
rmpath(genpath('./'));