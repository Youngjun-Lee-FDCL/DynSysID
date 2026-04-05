clear; clc; close all;

%% Add path
rootPath = genpath('./');
addpath(rootPath);

% GPML startup if available
if exist('startup', 'file') == 2
    try
        startup;
    catch
    end
end

rng(1);

%% =========================
% User-editable settings
%% =========================

% Example generator
% exampleFcn = @generate_nonlinear_msd_example;
% exampleFcn = @generate_frigola_benchmark_example;
% exampleFcn = @generate_linear_msd_example;
exampleFcn = @generate_nonlinear_twotank_example;
% exampleFcn = @generate_linear_mimo3_example;
% exampleFcn = @generate_nonlinear_cstr_PID_example;
% exampleFcn = @generate_nonlinear_aircraft43_PID_example;
% exampleFcn = @generate_vgpssm_easy_example;

% Dataset sizes
numExpTrain = 4;
numExpEval  = 5;

% Evaluation mode
evalMode = 'full';   % 'val' or 'full'

% N4SID setup
orders = 4;

optN4 = n4sidOptions;
optN4.Focus        = 'simulation';
optN4.N4Weight     = 'all';
optN4.N4Horizon    = 'auto';
optN4.InitialState = 'auto';

% Pseudo-state smoothing setup
P0Scale = 1.0;

% GP residual model setup
gpMaxIter = 120;
gpJitterY = 1e-8;

%% =========================
% Generate training dataset
%% =========================
dataTrain = exampleFcn(numExpTrain);

modelName = dataTrain.modelName;
Ts        = dataTrain.Ts;

[u0Raw, y0Raw] = force_2d_io(dataTrain.uCell{1}, dataTrain.yCell{1});
ny = size(y0Raw,2);
nu = size(u0Raw,2);

fprintf('\n=== Training dataset summary ===\n');
fprintf('Model name              : %s\n', modelName);
fprintf('Number of experiments   : %d\n', dataTrain.numExp);
fprintf('Input dimension         : %d\n', nu);
fprintf('Output dimension        : %d\n', ny);

%% =========================
% Build estimation / validation data
%% =========================
[zEstMerged, zEstCell, zValCell, yTrueHeldOut] = build_multi_exp_iddata(dataTrain, Ts);

%% =========================
% Estimate N4SID models
%% =========================
numOrders = numel(orders);
models = cell(numOrders,1);
fitSim_mean = zeros(numOrders,1);

for i = 1:numOrders
    nx = orders(i);
    sys_i = n4sid(zEstMerged, nx, optN4);
    models{i} = sys_i;

    ySimCell = cell(dataTrain.numExp,1);

    for e = 1:dataTrain.numExp
        x0est = findstates(sys_i, zValCell{e}, Inf);
        simOpt = simOptions('InitialCondition', x0est);
        ySimCell{e} = sim(sys_i, zValCell{e}, simOpt).OutputData;
    end

    ySimAll = vertcat(ySimCell{:});
    metrics_i = compute_fit_rmse(yTrueHeldOut, ySimAll);
    fitSim_mean(i) = metrics_i.fit;
end

%% =========================
% Pick best order
%% =========================
[~, bestIdx] = max(fitSim_mean);
bestOrder = orders(bestIdx);
sysN4 = models{bestIdx};

fprintf('\n=== N4SID summary ===\n');
fprintf('Best order              : %d\n', bestOrder);

A = sysN4.A;
B = sysN4.B;
C = sysN4.C;
D = sysN4.D;
nx = size(A,1);

%% =========================
% Build residual GP training data
%% =========================
XgpAll = [];
YgpAll = [];

opt.useResidualThreshold = true;
opt.residualQuantile = 0.7;   % 상위 30%만 사용

opt.useUniformSubsample = true;
opt.maxSamples = 2000;
opt.minSamples = 300;

for e = 1:dataTrain.numExp
    [uEst, yEst] = force_2d_io(dataTrain.uEstCell{e}, dataTrain.yEstCell{e});
    zEst = zEstCell{e};

    x0Est = findstates(sysN4, zEst, Inf);
    xSmEst = estimate_state_trajectory_rts(sysN4, uEst, yEst, x0Est, P0Scale);

    [Xgp_e, Ygp_e, idxKeep] = build_residual_dataset_selected(xSmEst, uEst, A, B, opt);

    XgpAll = [XgpAll; Xgp_e];
    YgpAll = [YgpAll; Ygp_e];
end

fprintf('\n=== Residual GP training data ===\n');
fprintf('Input dimension         : %d\n', size(XgpAll,2));
fprintf('Target dimension        : %d\n', size(YgpAll,2));
fprintf('Number of samples       : %d\n', size(XgpAll,1));

%% =========================
% Train residual GP models with GPML
%% =========================
gpModel = train_residual_gp_gpml_fast(XgpAll, YgpAll, gpMaxIter, gpJitterY);

%% =========================
% Generate test dataset
%% =========================
dataTest = exampleFcn(numExpEval);

fprintf('\n=== Test dataset summary ===\n');
fprintf('Number of experiments   : %d\n', dataTest.numExp);

%% =========================
% Evaluate on multiple test cases
%% =========================
yTrueCell   = cell(numExpEval,1);
yN4Cell     = cell(numExpEval,1);
yHybridCell = cell(numExpEval,1);
tCell       = cell(numExpEval,1);

xN4Cell     = cell(numExpEval,1);
xHybridCell = cell(numExpEval,1);
xTrueCell   = cell(numExpEval,1);

fitN4_each      = zeros(numExpEval,1);
rmseN4_each     = zeros(numExpEval,1);
fitHybrid_each  = zeros(numExpEval,1);
rmseHybrid_each = zeros(numExpEval,1);

for e = 1:numExpEval
    [uEval, yEval, tEval] = get_eval_data(dataTest, Ts, e, evalMode);
    zEval = iddata(yEval, uEval, Ts);

    % Best-initialized comparison
    x0Eval = findstates(sysN4, zEval, Inf);

    % Baseline N4SID free-run
    simOpt = simOptions('InitialCondition', x0Eval);
    [yN4, ~, xN4] = sim(sysN4, zEval, simOpt);

    % Hybrid free-run
    [yHybrid, xHybrid] = rollout_n4sid_gp_fast(sysN4, gpModel, uEval, x0Eval);

    yTrueCell{e}   = yEval;
    yN4Cell{e}     = yN4.OutputData;
    yHybridCell{e} = yHybrid;
    tCell{e}       = tEval;

    xN4Cell{e}     = xN4;
    xHybridCell{e} = xHybrid;

    mN4 = compute_fit_rmse(yEval, yN4.OutputData);
    mHy = compute_fit_rmse(yEval, yHybrid);

    fitN4_each(e)      = mN4.fit;
    rmseN4_each(e)     = mN4.rmse;
    fitHybrid_each(e)  = mHy.fit;
    rmseHybrid_each(e) = mHy.rmse;

    xTrueCell{e} = extract_true_state_if_available(dataTest, e, evalMode, size(yEval,1));
end

fprintf('\n=== Test summary over %d cases ===\n', numExpEval);
fprintf('N4SID mean FIT          : %.4f %%\n', mean(fitN4_each));
fprintf('N4SID mean RMSE         : %.6f\n', mean(rmseN4_each));
fprintf('Hybrid mean FIT         : %.4f %%\n', mean(fitHybrid_each));
fprintf('Hybrid mean RMSE        : %.6f\n', mean(rmseHybrid_each));
fprintf('Hybrid better FIT in    : %d / %d cases\n', sum(fitHybrid_each > fitN4_each), numExpEval);
fprintf('Hybrid better RMSE in   : %d / %d cases\n', sum(rmseHybrid_each < rmseN4_each), numExpEval);

%% =========================
% Plot outputs
%% =========================
plot_output_comparison(tCell, yTrueCell, yN4Cell, yHybridCell, modelName, evalMode);

%% =========================
% Plot states
%% =========================
plot_state_comparison(tCell, xN4Cell, xHybridCell, xTrueCell, modelName, evalMode);

%% Remove path
rmpath(rootPath);

%% ========================================================================
% Local functions
%% ========================================================================

function [u, y] = force_2d_io(u, y)
if isvector(u), u = u(:); end
if isvector(y), y = y(:); end
end

function x = force_2d_state(x)
if isvector(x)
    x = x(:);
end
if size(x,1) < size(x,2)
    x = x.';
end
end

function [zEstMerged, zEstCell, zValCell, yTrueAll] = build_multi_exp_iddata(data, Ts)

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

function [uEval, yEval, tEval] = get_eval_data(data, Ts, expIdx, evalMode)

switch lower(evalMode)
    case 'val'
        [uEval, yEval] = force_2d_io(data.uValCell{expIdx}, data.yValCell{expIdx});
        if isfield(data, 'tCell') && isfield(data, 'idxValCell')
            tEval = data.tCell{expIdx}(data.idxValCell{expIdx});
        else
            tEval = (0:size(yEval,1)-1)' * Ts;
        end

    case 'full'
        [uEval, yEval] = force_2d_io(data.uCell{expIdx}, data.yCell{expIdx});
        if isfield(data, 'tCell')
            tEval = data.tCell{expIdx};
        else
            tEval = (0:size(yEval,1)-1)' * Ts;
        end

    otherwise
        error('evalMode must be ''val'' or ''full''.');
end
end

function metrics = compute_fit_rmse(yTrue, yHat)

yTrue = double(yTrue);
yHat  = double(yHat);

[~, ny] = size(yTrue);

rmse_each = sqrt(mean((yTrue - yHat).^2, 1));
fit_each  = zeros(1, ny);

for k = 1:ny
    denom = norm(yTrue(:,k) - mean(yTrue(:,k)));
    if denom < eps
        fit_each(k) = NaN;
    else
        fit_each(k) = 100 * (1 - norm(yTrue(:,k) - yHat(:,k)) / denom);
    end
end

metrics.rmse_each = rmse_each;
metrics.fit_each  = fit_each;
metrics.rmse = mean(rmse_each, 'omitnan');
metrics.fit  = mean(fit_each, 'omitnan');
end

function xSmooth = estimate_state_trajectory_rts(sysN4, u, y, x0, P0Scale)
% RTS smoother for approximate linear-Gaussian model
%
% x_{t+1} = A x_t + B u_t + w_t, w ~ N(0,Q)
% y_t     = C x_t + D u_t + v_t, v ~ N(0,R)

[u, y] = force_2d_io(u, y);

A = sysN4.A;
B = sysN4.B;
C = sysN4.C;
D = sysN4.D;

nx = size(A,1);
T  = size(y,1);

R = sysN4.NoiseVariance;
R = regularize_cov(R, 1e-8);

try
    K = sysN4.K;
catch
    error('sysN4.K could not be accessed.');
end

Q = K * R * K.';
Q = regularize_cov(Q, 1e-10);

P0 = P0Scale * eye(nx);

xPred = zeros(nx, T);
xFilt = zeros(nx, T);
PPred = zeros(nx, nx, T);
PFilt = zeros(nx, nx, T);

I = eye(nx);

% Initial prior for x_1
xPred(:,1) = x0(:);
PPred(:,:,1) = P0;

for t = 1:T
    ut = u(t,:).';
    yt = y(t,:).';

    S = C * PPred(:,:,t) * C.' + R;
    S = regularize_cov(S, 1e-12);

    Kgain = PPred(:,:,t) * C.' / S;
    innov = yt - (C * xPred(:,t) + D * ut);

    xFilt(:,t) = xPred(:,t) + Kgain * innov;
    PFilt(:,:,t) = (I - Kgain * C) * PPred(:,:,t);
    PFilt(:,:,t) = 0.5 * (PFilt(:,:,t) + PFilt(:,:,t).');

    if t < T
        xPred(:,t+1) = A * xFilt(:,t) + B * ut;
        PPred(:,:,t+1) = A * PFilt(:,:,t) * A.' + Q;
        PPred(:,:,t+1) = 0.5 * (PPred(:,:,t+1) + PPred(:,:,t+1).');
    end
end

% RTS backward pass
xSmooth = zeros(nx, T);
PSmooth = zeros(nx, nx, T); %#ok<NASGU>

xSmooth(:,T) = xFilt(:,T);
PSmooth(:,:,T) = PFilt(:,:,T);

for t = T-1:-1:1
    J = PFilt(:,:,t) * A.' / PPred(:,:,t+1);
    xSmooth(:,t) = xFilt(:,t) + J * (xSmooth(:,t+1) - xPred(:,t+1));
    PSmooth(:,:,t) = PFilt(:,:,t) + J * (PSmooth(:,:,t+1) - PPred(:,:,t+1)) * J.';
    PSmooth(:,:,t) = 0.5 * (PSmooth(:,:,t) + PSmooth(:,:,t).');
end
end

function [Xgp, Ygp] = build_residual_dataset(xSmooth, u, A, B)
% Build residual learning dataset:
% input  z_t = [x_t, u_t]
% target d_t = x_{t+1} - A x_t - B u_t

[u, ~] = force_2d_io(u, zeros(size(u,1),1)); %#ok<ASGLU>
[nx, T] = size(xSmooth);
nu = size(u,2);

N = T - 1;
Xgp = zeros(N, nx + nu);
Ygp = zeros(N, nx);

for t = 1:N
    xt = xSmooth(:,t);
    ut = u(t,:).';
    xLinNext = A * xt + B * ut;
    dxt = xSmooth(:,t+1) - xLinNext;

    Xgp(t,:) = [xt.', ut.'];
    Ygp(t,:) = dxt.';
end
end

function gpModel = train_residual_gp_gpml_fast(X, Y, maxIter, jitterY)
% Independent exact GP for each state component
% Store alpha so that predictive mean is computed by k_*' * alpha

X = double(X);
Y = double(Y);

[~, Din] = size(X);
nx = size(Y,2);

muX = mean(X, 1);
stdX = std(X, 0, 1);
stdX(stdX < 1e-12) = 1;

Xn = (X - muX) ./ stdX;

meanfunc = @meanZero;
covfunc  = @covSEard;
likfunc  = @likGauss;
inffunc  = @infGaussLik;

gpList = cell(nx,1);

for j = 1:nx
    yj = Y(:,j);
    muY = mean(yj);
    stdY = std(yj);
    if stdY < 1e-12
        stdY = 1;
    end

    Yn = (yj - muY) ./ stdY;

    hyp = struct();
    hyp.mean = [];
    hyp.cov  = [zeros(Din,1); 0];
    hyp.lik  = log(max(std(Yn) * 0.1, 1e-3) + jitterY);

    hyp = minimize(hyp, @gp, -maxIter, inffunc, meanfunc, covfunc, likfunc, Xn, Yn);

    % Precompute alpha = (K + sn2 I)^(-1) y
    K = feval(covfunc, hyp.cov, Xn);
    sn2 = exp(2*hyp.lik);
    Ktilde = K + sn2 * eye(size(K));

    Ktilde = 0.5 * (Ktilde + Ktilde.');
    L = chol(Ktilde, 'lower');
    alpha = L' \ (L \ Yn);

    gpList{j} = struct( ...
        'hyp', hyp, ...
        'muY', muY, ...
        'stdY', stdY, ...
        'Xn', Xn, ...
        'Yn', Yn, ...
        'alpha', alpha, ...
        'L', L, ...
        'meanfunc', meanfunc, ...
        'covfunc', covfunc, ...
        'likfunc', likfunc, ...
        'inffunc', inffunc);
end

gpModel = struct();
gpModel.muX = muX;
gpModel.stdX = stdX;
gpModel.gpList = gpList;
gpModel.nx = nx;
gpModel.Din = Din;
end

function delta = predict_residual_gp_fast(gpModel, z)
% Fast predictive mean using k_*' * alpha
% z : [1 x Din]

z = double(z(:).');
zn = (z - gpModel.muX) ./ gpModel.stdX;

nx = gpModel.nx;
delta = zeros(nx,1);

for j = 1:nx
    gpj = gpModel.gpList{j};

    % kstar: [N x 1]
    kstar = feval(gpj.covfunc, gpj.hyp.cov, gpj.Xn, zn);

    % meanZero assumed
    muN = kstar.' * gpj.alpha;

    delta(j) = muN * gpj.stdY + gpj.muY;
end
end

function [yHat, xHat] = rollout_n4sid_gp_fast(sysN4, gpModel, u, x0)
% Free-run rollout of hybrid model:
% x_{t+1} = A x_t + B u_t + GP([x_t, u_t])
% y_t     = C x_t + D u_t

A = sysN4.A;
B = sysN4.B;
C = sysN4.C;
D = sysN4.D;

[u, ~] = force_2d_io(u, zeros(size(u,1),1)); %#ok<ASGLU>
T = size(u,1);
nx = size(A,1);
ny = size(C,1);

xHat = zeros(T, nx);
yHat = zeros(T, ny);

xHat(1,:) = x0(:).';

for t = 1:T
    xt = xHat(t,:).';
    ut = u(t,:).';

    yHat(t,:) = (C * xt + D * ut).';

    if t < T
        z = [xt.', ut.'];
        delta = predict_residual_gp_fast(gpModel, z);
        xNext = A * xt + B * ut + delta;
        xHat(t+1,:) = xNext.';
    end
end
end

function xTrue = extract_true_state_if_available(data, expIdx, evalMode, Treq)

xTrue = [];

if strcmpi(evalMode, 'val')
    if isfield(data, 'xValCell') && numel(data.xValCell) >= expIdx && ~isempty(data.xValCell{expIdx})
        xTmp = force_2d_state(data.xValCell{expIdx});
        if size(xTmp,1) == Treq
            xTrue = xTmp;
            return;
        end
    end

    if isfield(data, 'xCell') && numel(data.xCell) >= expIdx && ~isempty(data.xCell{expIdx})
        xTmp = force_2d_state(data.xCell{expIdx});

        if isfield(data, 'idxValCell') && numel(data.idxValCell) >= expIdx && ~isempty(data.idxValCell{expIdx})
            idxVal = data.idxValCell{expIdx};
            try
                xTmp2 = xTmp(idxVal, :);
                if size(xTmp2,1) == Treq
                    xTrue = xTmp2;
                end
            catch
                xTrue = [];
            end
        end
    end

else
    if isfield(data, 'xCell') && numel(data.xCell) >= expIdx && ~isempty(data.xCell{expIdx})
        xTmp = force_2d_state(data.xCell{expIdx});
        if size(xTmp,1) == Treq
            xTrue = xTmp;
        end
    end
end
end

function plot_output_comparison(tCell, yTrueCell, yN4Cell, yHybridCell, modelName, evalMode)

numExp = numel(tCell);
ny = size(yTrueCell{1},2);

for e = 1:numExp
    figure('Name', sprintf('%s | %s | Output | Exp %d', modelName, evalMode, e));
    tiledlayout(ny,1);

    t = tCell{e};
    yTrue = yTrueCell{e};
    yN4 = yN4Cell{e};
    yHybrid = yHybridCell{e};

    for k = 1:ny
        nexttile;
        plot(t, yTrue(:,k), 'k', 'LineWidth', 1.5); hold on;
        plot(t, yN4(:,k), 'b--', 'LineWidth', 1.2);
        plot(t, yHybrid(:,k), 'r-', 'LineWidth', 1.2);
        grid on;
        ylabel(sprintf('y_%d', k));

        if k == 1
            title(sprintf('%s | %s | Output comparison | Exp %d', modelName, evalMode, e));
            legend('Measured', 'N4SID free-run', 'N4SID + residual GP', 'Location', 'best');
        end
    end
    xlabel('Time [s]');
end
end

function plot_state_comparison(tCell, xN4Cell, xHybridCell, xTrueCell, modelName, evalMode)

numExp = numel(tCell);
nx = size(xN4Cell{1},2);

for e = 1:numExp
    figure('Name', sprintf('%s | %s | States | Exp %d', modelName, evalMode, e));
    tiledlayout(nx,1);

    t = tCell{e};
    xN4 = xN4Cell{e};
    xHybrid = xHybridCell{e};
    xTrue = xTrueCell{e};

    for i = 1:nx
        nexttile;
        plot(t, xN4(:,i), 'b-', 'LineWidth', 1.1); hold on;
        plot(t, xHybrid(:,i), 'r--', 'LineWidth', 1.2);

        if ~isempty(xTrue) && size(xTrue,2) >= i
            plot(t, xTrue(:,i), 'k', 'LineWidth', 1.0);
        end

        grid on;
        ylabel(sprintf('x_%d', i));

        if i == 1
            title(sprintf('%s | %s | State comparison | Exp %d', modelName, evalMode, e));
            if ~isempty(xTrue) && size(xTrue,2) >= i
                legend('N4SID free-run', 'N4SID + residual GP', 'True', 'Location', 'best');
            else
                legend('N4SID free-run', 'N4SID + residual GP', 'Location', 'best');
            end
        end
    end
    xlabel('Time [s]');
end
end

function S = regularize_cov(S, jitter)
S = 0.5 * (S + S.');
[~, p] = chol(S);
if p ~= 0
    S = S + jitter * eye(size(S));
end
end

function [Xgp, Ygp, idxKeep] = build_residual_dataset_selected( ...
    xSmooth, u, A, B, opt)
% SELECTED residual dataset builder
%
% opt fields:
%   .useResidualThreshold
%   .residualQuantile (e.g., 0.7)
%   .useUniformSubsample
%   .maxSamples
%   .minSamples

    [u, ~] = force_2d_io(u, zeros(size(u,1),1)); %#ok<ASGLU>
    [nx, T] = size(xSmooth);
    nu = size(u,2);

    N = T - 1;

    Xall = zeros(N, nx + nu);
    Yall = zeros(N, nx);
    resNorm = zeros(N,1);

    % ------------------------------------------------------------
    % Build full dataset
    % ------------------------------------------------------------
    for t = 1:N
        xt = xSmooth(:,t);
        ut = u(t,:).';

        xLinNext = A * xt + B * ut;
        dxt = xSmooth(:,t+1) - xLinNext;

        Xall(t,:) = [xt.', ut.'];
        Yall(t,:) = dxt.';

        resNorm(t) = norm(dxt);
    end

    idxKeep = (1:N).';

    % ------------------------------------------------------------
    % 1) Residual-based selection
    % ------------------------------------------------------------
    if opt.useResidualThreshold
        th = quantile(resNorm, opt.residualQuantile);

        idxKeep = find(resNorm >= th);

        % safety: 너무 적으면 보완
        if numel(idxKeep) < opt.minSamples
            [~, idxSort] = sort(resNorm, 'descend');
            idxKeep = idxSort(1:opt.minSamples);
        end
    end

    % ------------------------------------------------------------
    % 2) Uniform subsampling (coverage 유지)
    % ------------------------------------------------------------
    if opt.useUniformSubsample
        if numel(idxKeep) > opt.maxSamples
            step = floor(numel(idxKeep) / opt.maxSamples);
            idxKeep = idxKeep(1:step:end);
        end
    end

    % ------------------------------------------------------------
    % Final dataset
    % ------------------------------------------------------------
    Xgp = Xall(idxKeep,:);
    Ygp = Yall(idxKeep,:);
end