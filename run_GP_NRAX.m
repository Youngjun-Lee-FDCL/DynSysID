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
% exampleFcn = @generate_linear_mimo3_example;   % <- MIMO example
% exampleFcn = @generate_nonlinear_cstr_PID_example;
exampleFcn = @generate_nonlinear_aircraft43_PID_example;
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
% X(k,:) uses past outputs of all channels + past inputs of all channels
% Y(k,:) = current outputs of all channels
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
% One-step prediction metrics
%% =========================
rmse1_each = sqrt(mean((Yte - muTe).^2, 1));
fit1_each  = zeros(1, ny);
for j = 1:ny
    fit1_each(j) = 100 * (1 - norm(Yte(:,j) - muTe(:,j)) / norm(Yte(:,j) - mean(Yte(:,j))));
end

rmse1 = mean(rmse1_each);
fit1  = mean(fit1_each);

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

rmseFree_each = sqrt(mean((Yte - yHatFree).^2, 1));
fitFree_each  = zeros(1, ny);
for j = 1:ny
    fitFree_each(j) = 100 * (1 - norm(Yte(:,j) - yHatFree(:,j)) / norm(Yte(:,j) - mean(Yte(:,j))));
end

rmseFree = mean(rmseFree_each);
fitFree  = mean(fitFree_each);

%% =========================
% Plot
%% =========================
tTe = t(idxTe);

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

figure('Color','w');
for j = 1:ny
    subplot(ny,1,j);

    upper = muTe(:,j) + 2*sqrt(max(s2Te(:,j),0));
    lower = muTe(:,j) - 2*sqrt(max(s2Te(:,j),0));

    fill([tTe; flipud(tTe)], [upper; flipud(lower)], ...
        [0.85 0.90 1.00], 'EdgeColor', 'none'); hold on;
    plot(tTe, Yte(:,j), 'k', 'LineWidth', 1.2);
    plot(tTe, muTe(:,j), 'r--', 'LineWidth', 1.4);
    grid on;
    xlabel('Time (s)');
    ylabel(sprintf('y_%d', j));
    legend('95% band', 'True', 'GP-NARX one-step', 'Location', 'best');
    title(sprintf('One-step | output %d | RMSE = %.4f, FIT = %.2f%%', ...
        j, rmse1_each(j), fit1_each(j)));
end
sgtitle([modelName, ' | One-step prediction']);

figure('Color','w');
for j = 1:ny
    subplot(ny,1,j);
    plot(tTe, Yte(:,j), 'k', 'LineWidth', 1.2); hold on;
    plot(tTe, yHatFree(:,j), 'b--', 'LineWidth', 1.4);
    grid on;
    xlabel('Time (s)');
    ylabel(sprintf('y_%d', j));
    legend('True', 'GP-NARX free-run', 'Location', 'best');
    title(sprintf('Free-run | output %d | RMSE = %.4f, FIT = %.2f%%', ...
        j, rmseFree_each(j), fitFree_each(j)));
end
sgtitle([modelName, ' | Free-run simulation']);

fprintf('\n');
fprintf('Model             : %s\n', modelName);
fprintf('Outputs           : %d\n', ny);
fprintf('Inputs            : %d\n', nu);
fprintf('Mean One-step RMSE: %.6f\n', rmse1);
fprintf('Mean One-step FIT : %.2f %%\n', fit1);
fprintf('Mean Free-run RMSE: %.6f\n', rmseFree);
fprintf('Mean Free-run FIT : %.2f %%\n', fitFree);
for j = 1:ny
    fprintf('  y_%d -> one-step RMSE %.6f, FIT %.2f %% | free-run RMSE %.6f, FIT %.2f %%\n', ...
        j, rmse1_each(j), fit1_each(j), rmseFree_each(j), fitFree_each(j));
end
fprintf('\n');

%% remove path
rmpath(genpath('./'));