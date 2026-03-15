clear; clc; close all;

rng(1);

addpath(genpath('./'));
%% =========================
% Select example generator
%% =========================
% exampleFcn = @generate_toy_nonlinear_example;
% exampleFcn = @generate_linear_msd_example;
% exampleFcn = @generate_nonlinear_msd_example;
% exampleFcn = @generate_frigola_benchmark_example;
% exampleFcn = @generate_linear_mimo3_example;   % <- MIMO example
% exampleFcn = @generate_linear_mimo3_PID_example;   % <- MIMO example
exampleFcn = @generate_nonlinear_twotank_example;
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
    tVal = data.t(data.idxVal);
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
opt = nlarxOptions;
opt.SearchOptions.MaxIterations = 50;
opt.Focus = 'simulation';

%% =========================
% Estimate NLARX model
%% =========================
% For MIMO:
%   na : ny-by-ny
%   nb : ny-by-nu
%   nk : ny-by-nu
% For SISO:
%   scalars are also fine
sysNLARX = nlarx(zEst, [na nb nk], outputFcn, opt);

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
opt = simOptions;
opt.InitialCondition = x0;

ySimObj = sim(sysNLARX, zVal, opt);
ySim = ySimObj.OutputData;     % Nval x ny

%% =========================
% Metrics (per channel + average)
%% =========================
rmse1_each    = sqrt(mean((yTrue - yPred).^2, 1));
rmseFree_each = sqrt(mean((yTrue - ySim ).^2, 1));

fit1_each    = zeros(1, ny);
fitFree_each = zeros(1, ny);

for j = 1:ny
    denom1 = norm(yTrue(:,j) - mean(yTrue(:,j)));
    if denom1 < 1e-12
        fit1_each(j) = NaN;
        fitFree_each(j) = NaN;
    else
        fit1_each(j)    = 100 * (1 - norm(yTrue(:,j) - yPred(:,j)) / denom1);
        fitFree_each(j) = 100 * (1 - norm(yTrue(:,j) - ySim(:,j))  / denom1);
    end
end

rmse1    = mean(rmse1_each, 'omitnan');
rmseFree = mean(rmseFree_each, 'omitnan');
fit1     = mean(fit1_each, 'omitnan');
fitFree  = mean(fitFree_each, 'omitnan');

%% =========================
% Plot measured outputs
%% =========================
figure('Color','w');
for j = 1:ny
    subplot(ny,1,j);
    plot(t, y(:,j), 'LineWidth', 1.2);
    grid on;
    xlabel('Time (s)');
    ylabel(sprintf('y_%d', j));
    if j == 1
        title(['Measured output - ', modelName]);
    end
end

%% =========================
% Plot one-step prediction
%% =========================
figure('Color','w');
for j = 1:ny
    subplot(ny,1,j);
    plot(tVal, yTrue(:,j), 'k', 'LineWidth', 1.2); hold on;
    plot(tVal, yPred(:,j), 'r--', 'LineWidth', 1.4);
    grid on;
    xlabel('Time (s)');
    ylabel(sprintf('y_%d', j));
    legend('True', 'NLARX one-step', 'Location', 'best');
    title(sprintf('One-step | y_%d | RMSE = %.4f, FIT = %.2f%%', ...
        j, rmse1_each(j), fit1_each(j)));
end
sgtitle([modelName, ' | One-step prediction']);

%% =========================
% Plot free-run simulation
%% =========================
figure('Color','w');
for j = 1:ny
    subplot(ny,1,j);
    plot(tVal, yTrue(:,j), 'k', 'LineWidth', 1.2); hold on;
    plot(tVal, ySim(:,j), 'b--', 'LineWidth', 1.4);
    grid on;
    xlabel('Time (s)');
    ylabel(sprintf('y_%d', j));
    legend('True', 'NLARX free-run', 'Location', 'best');
    title(sprintf('Free-run | y_%d | RMSE = %.4f, FIT = %.2f%%', ...
        j, rmseFree_each(j), fitFree_each(j)));
end
sgtitle([modelName, ' | Free-run simulation']);

%% =========================
% Plot errors
%% =========================
figure('Color','w');
for j = 1:ny
    subplot(ny,2,2*j-1);
    plot(tVal, yTrue(:,j) - yPred(:,j), 'LineWidth', 1.2);
    grid on;
    xlabel('Time (s)');
    ylabel(sprintf('e_{1step,%d}', j));
    title(sprintf('One-step error | y_%d', j));

    subplot(ny,2,2*j);
    plot(tVal, yTrue(:,j) - ySim(:,j), 'LineWidth', 1.2);
    grid on;
    xlabel('Time (s)');
    ylabel(sprintf('e_{free,%d}', j));
    title(sprintf('Free-run error | y_%d', j));
end

%% =========================
% Print metrics
%% =========================
fprintf('\n');
fprintf('Model              : %s\n', modelName);
fprintf('NLARX type         : %s\n', class(outputFcn));
fprintf('Number of inputs   : %d\n', nu);
fprintf('Number of outputs  : %d\n', ny);
fprintf('Mean One-step RMSE : %.6f\n', rmse1);
fprintf('Mean One-step FIT  : %.2f %%\n', fit1);
fprintf('Mean Free-run RMSE : %.6f\n', rmseFree);
fprintf('Mean Free-run FIT  : %.2f %%\n', fitFree);

for j = 1:ny
    fprintf('  y_%d -> one-step RMSE %.6f, FIT %.2f %% | free-run RMSE %.6f, FIT %.2f %%\n', ...
        j, rmse1_each(j), fit1_each(j), rmseFree_each(j), fitFree_each(j));
end
fprintf('\n');

%% Remove path
rmpath(genpath('./'));