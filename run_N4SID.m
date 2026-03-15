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
% exampleFcn = @generate_linear_mimo3_example;   % <- MIMO example
% exampleFcn = @generate_linear_mimo3_PID_example;   % <- MIMO example
% exampleFcn = @generate_nonlinear_twotank_example;
exampleFcn = @generate_nonlinear_aircraft43_PID_example;
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
opt = n4sidOptions;
opt.Focus = 'simulation';
opt.N4Weight = 'all';      % alternatives: 'MOESP', 'SSARX', 'auto'
opt.N4Horizon = 'auto';
opt.InitialState = 'auto';

%% =========================
% Estimate models of multiple orders
%% =========================
models = cell(numel(orders),1);

fitPred_each = zeros(numel(orders), ny);
fitSim_each  = zeros(numel(orders), ny);
rmsePred_each = zeros(numel(orders), ny);
rmseSim_each  = zeros(numel(orders), ny);

fitPred_mean = zeros(numel(orders),1);
fitSim_mean  = zeros(numel(orders),1);
rmsePred_mean = zeros(numel(orders),1);
rmseSim_mean  = zeros(numel(orders),1);

for i = 1:numel(orders)
    nx = orders(i);

    models{i} = n4sid(zEst, nx, opt);

    % one-step prediction
    yPredObj = predict(models{i}, zVal, 1);
    yPred = yPredObj.OutputData;

    % free-run simulation
    ySimObj = sim(models{i}, zVal);
    ySim = ySimObj.OutputData;

    for j = 1:ny
        rmsePred_each(i,j) = sqrt(mean((yTrue(:,j) - yPred(:,j)).^2));
        rmseSim_each(i,j)  = sqrt(mean((yTrue(:,j) - ySim(:,j)).^2));

        denom = norm(yTrue(:,j) - mean(yTrue(:,j)));
        if denom < 1e-12
            fitPred_each(i,j) = NaN;
            fitSim_each(i,j)  = NaN;
        else
            fitPred_each(i,j) = 100 * (1 - norm(yTrue(:,j) - yPred(:,j)) / denom);
            fitSim_each(i,j)  = 100 * (1 - norm(yTrue(:,j) - ySim(:,j))  / denom);
        end
    end

    fitPred_mean(i)  = mean(fitPred_each(i,:), 'omitnan');
    fitSim_mean(i)   = mean(fitSim_each(i,:), 'omitnan');
    rmsePred_mean(i) = mean(rmsePred_each(i,:), 'omitnan');
    rmseSim_mean(i)  = mean(rmseSim_each(i,:), 'omitnan');
end

%% =========================
% Pick best order based on mean free-run FIT
%% =========================
[~, bestIdx] = max(fitSim_mean);
bestOrder = orders(bestIdx);
sysN4 = models{bestIdx};

fprintf('\n');
fprintf('Model                   : %s\n', modelName);
fprintf('Number of inputs        : %d\n', nu);
fprintf('Number of outputs       : %d\n', ny);
fprintf('Best order              : %d\n', bestOrder);
fprintf('Best mean one-step RMSE : %.6f\n', rmsePred_mean(bestIdx));
fprintf('Best mean one-step FIT  : %.2f %%\n', fitPred_mean(bestIdx));
fprintf('Best mean free-run RMSE : %.6f\n', rmseSim_mean(bestIdx));
fprintf('Best mean free-run FIT  : %.2f %%\n', fitSim_mean(bestIdx));
fprintf('\n');

%% =========================
% Final one-step / free-run using selected model
%% =========================
x0 = findstates(sysN4, zVal);
opt = simOptions('InitialCondition', x0);

yPredObj = predict(sysN4, zVal, 1, opt);
yPred = yPredObj.OutputData;

ySimObj = sim(sysN4, zVal, opt);
ySim = ySimObj.OutputData;

rmsePred_best_each = rmsePred_each(bestIdx,:);
rmseSim_best_each  = rmseSim_each(bestIdx,:);
fitPred_best_each  = fitPred_each(bestIdx,:);
fitSim_best_each   = fitSim_each(bestIdx,:);

%% =========================
% Plot measured output
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
% Plot order selection results
%% =========================
figure('Color','w');
yyaxis left;
plot(orders, fitPred_mean, 'o-', 'LineWidth', 1.2); hold on;
plot(orders, fitSim_mean,  's-', 'LineWidth', 1.2);
ylabel('Mean FIT (%)');

yyaxis right;
plot(orders, rmsePred_mean, 'o--', 'LineWidth', 1.0);
plot(orders, rmseSim_mean,  's--', 'LineWidth', 1.0);
ylabel('Mean RMSE');

grid on;
xlabel('Model order');
title([modelName, ' - N4SID order scan']);
legend('Mean one-step FIT', 'Mean free-run FIT', ...
       'Mean one-step RMSE', 'Mean free-run RMSE', ...
       'Location', 'best');

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
    legend('True', 'N4SID one-step', 'Location', 'best');
    title(sprintf('One-step | y_%d | order=%d | RMSE=%.4f | FIT=%.2f%%', ...
        j, bestOrder, rmsePred_best_each(j), fitPred_best_each(j)));
end
sgtitle([modelName, ' | N4SID one-step prediction']);

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
    legend('True', 'N4SID free-run', 'Location', 'best');
    title(sprintf('Free-run | y_%d | order=%d | RMSE=%.4f | FIT=%.2f%%', ...
        j, bestOrder, rmseSim_best_each(j), fitSim_best_each(j)));
end
sgtitle([modelName, ' | N4SID free-run simulation']);

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
% Optional: model info
%% =========================
disp(sysN4);

%% =========================
% Print per-channel metrics
%% =========================
for j = 1:ny
    fprintf('y_%d -> one-step RMSE %.6f, FIT %.2f %% | free-run RMSE %.6f, FIT %.2f %%\n', ...
        j, rmsePred_best_each(j), fitPred_best_each(j), ...
        rmseSim_best_each(j),  fitSim_best_each(j));
end
fprintf('\n');

%% Remove path
rmpath(genpath('./'));