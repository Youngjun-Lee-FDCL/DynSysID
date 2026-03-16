function data = generate_nonlinear_twotank_example(numExp)
% Generate GP-NARX training/test data for nonlinear two-tank system
%
% Multi-experiment version:
%   Each experiment is simulated independently.
%   Outputs are stored in cell arrays.
%
% Continuous-time model:
%   dx1/dt = (1/A1) * (k*u - a1*sqrt(2*g*x1))
%   dx2/dt = (1/A2) * (a1*sqrt(2*g*x1) - a2*sqrt(2*g*x2))
%   y      = x2 + noise
%
% State:
%   x1 : upper tank level
%   x2 : lower tank level

    if nargin < 1 || isempty(numExp)
        numExp = 1;
    end

    rng(1);

    %% Simulation settings
    Ts = 0.2;
    N  = 1000;
    t  = (0:N-1)' * Ts;

    %% True parameters
    A1 = 0.5;
    A2 = 0.25;
    a1 = 0.02;
    a2 = 0.015;
    k  = 0.005;
    g  = 9.81;

    sigmaY = 0.0001;

    %% NARX settings
    na = 10;
    nb = 10;
    nk = 1;

    %% Preallocate cells
    uCell      = cell(numExp,1);
    yCell      = cell(numExp,1);
    xCell      = cell(numExp,1);
    tCell      = cell(numExp,1);

    XCell      = cell(numExp,1);
    YCell      = cell(numExp,1);
    idxMapCell = cell(numExp,1);

    XtrCell    = cell(numExp,1);
    YtrCell    = cell(numExp,1);
    XteCell    = cell(numExp,1);
    YteCell    = cell(numExp,1);
    idxTeCell  = cell(numExp,1);

    uEstCell   = cell(numExp,1);
    yEstCell   = cell(numExp,1);
    uValCell   = cell(numExp,1);
    yValCell   = cell(numExp,1);
    idxValCell = cell(numExp,1);

    expInfo = repmat(struct(), numExp, 1);

    %% Run multiple independent experiments
    for e = 1:numExp
        % ------------------------------------------------------------
        % Experiment-specific input
        % ------------------------------------------------------------
        amp1   = 0.7 + 0.3*rand;
        amp2   = 0.4 + 0.2*rand;
        freq1  = 0.003 * (0.85 + 0.30*rand);
        freq2  = 0.011 * (0.85 + 0.30*rand);
        phase2 = 2*pi*rand;
        biasU  = 2.8 + 0.5*rand;
        noiseU = 0.15 + 0.10*rand;

        u = biasU ...
          + amp1*sin(2*pi*freq1*t) ...
          + amp2*sin(2*pi*freq2*t + phase2) ...
          + noiseU*randn(N,1);

        % Add a few random step-like perturbations
        nSteps = 3 + randi(2);
        stepTimes = sort(randsample((50:N-100), nSteps));
        for kk = 1:nSteps
            u(stepTimes(kk):end) = u(stepTimes(kk):end) + 0.2*randn;
        end

        % Pump input should be nonnegative
        u = max(u, 0);

        % ------------------------------------------------------------
        % Slightly perturbed initial condition
        % ------------------------------------------------------------
        x0 = [ ...
            max(0.0, 0.2 + 0.03*randn); ...
            max(0.0, 0.1 + 0.02*randn)];

        % ------------------------------------------------------------
        % Simulate
        % ------------------------------------------------------------
        x = zeros(2, N);   % x(1,:) = upper tank level, x(2,:) = lower tank level
        y = zeros(N, 1);

        x(:,1) = x0;

        for kk = 1:N-1
            x(:,kk+1) = rk4_step_twotank(x(:,kk), u(kk), Ts, A1, A2, a1, a2, k, g);

            % Enforce nonnegative water levels
            x(:,kk+1) = max(x(:,kk+1), 0);

            y(kk) = x(2,kk) + sigmaY*randn;
        end
        y(N) = x(2,N) + sigmaY*randn;

        % ------------------------------------------------------------
        % Build NARX regressors
        % ------------------------------------------------------------
        [X, Y, idxMap] = build_narx_regressors(y, u, na, nb, nk);

        % ------------------------------------------------------------
        % Split
        % ------------------------------------------------------------
        Ndata = size(X,1);
        Ntr   = round(0.7*Ndata);

        % Full trajectory split index on original time axis
        NtrOriginal = round(0.7*N);

        % ------------------------------------------------------------
        % Store experiment data
        % ------------------------------------------------------------
        uCell{e} = u;
        yCell{e} = y;
        xCell{e} = x;
        tCell{e} = t;

        XCell{e}      = X;
        YCell{e}      = Y;
        idxMapCell{e} = idxMap;

        XtrCell{e}   = X(1:Ntr,:);
        YtrCell{e}   = Y(1:Ntr);
        XteCell{e}   = X(Ntr+1:end,:);
        YteCell{e}   = Y(Ntr+1:end);
        idxTeCell{e} = idxMap(Ntr+1:end);

        uEstCell{e}   = u(1:NtrOriginal,:);
        yEstCell{e}   = y(1:NtrOriginal,:);
        uValCell{e}   = u(NtrOriginal+1:end,:);
        yValCell{e}   = y(NtrOriginal+1:end,:);
        idxValCell{e} = (NtrOriginal+1:N).';

        expInfo(e).x0 = x0;
        expInfo(e).inputSettings = struct( ...
            'biasU',  biasU, ...
            'amp1',   amp1, ...
            'amp2',   amp2, ...
            'freq1',  freq1, ...
            'freq2',  freq2, ...
            'phase2', phase2, ...
            'noiseU', noiseU);
    end

    %% Pack data
    data = struct();
    data.modelName = sprintf('Nonlinear Two-Tank (%d experiments)', numExp);
    data.isMultiExperiment = (numExp > 1);
    data.numExp = numExp;

    data.Ts = Ts;
    data.t  = t;
    data.N  = N;

    data.na = na;
    data.nb = nb;
    data.nk = nk;

    data.uCell = uCell;
    data.yCell = yCell;
    data.xCell = xCell;
    data.tCell = tCell;

    data.XCell      = XCell;
    data.YCell      = YCell;
    data.idxMapCell = idxMapCell;

    data.XtrCell   = XtrCell;
    data.YtrCell   = YtrCell;
    data.XteCell   = XteCell;
    data.YteCell   = YteCell;
    data.idxTeCell = idxTeCell;

    data.uEstCell   = uEstCell;
    data.yEstCell   = yEstCell;
    data.uValCell   = uValCell;
    data.yValCell   = yValCell;
    data.idxValCell = idxValCell;

    data.params = struct( ...
        'A1', A1, ...
        'A2', A2, ...
        'a1', a1, ...
        'a2', a2, ...
        'k',  k, ...
        'g',  g, ...
        'sigmaY', sigmaY);

    data.expInfo = expInfo;

    %% Backward compatibility for single experiment
    if numExp == 1
        data.u = uCell{1};
        data.y = yCell{1};
        data.x = xCell{1};

        data.X = XCell{1};
        data.Y = YCell{1};

        data.Xtr = XtrCell{1};
        data.Ytr = YtrCell{1};
        data.Xte = XteCell{1};
        data.Yte = YteCell{1};
        data.idxTe = idxTeCell{1};

        data.uEst = uEstCell{1};
        data.yEst = yEstCell{1};
        data.uVal = uValCell{1};
        data.yVal = yValCell{1};
        data.idxVal = idxValCell{1};
    end
end

function xNext = rk4_step_twotank(x, u, Ts, A1, A2, a1, a2, k, g)
    k1 = f_twotank(x, u, A1, A2, a1, a2, k, g);
    k2 = f_twotank(x + 0.5*Ts*k1, u, A1, A2, a1, a2, k, g);
    k3 = f_twotank(x + 0.5*Ts*k2, u, A1, A2, a1, a2, k, g);
    k4 = f_twotank(x + Ts*k3,     u, A1, A2, a1, a2, k, g);

    xNext = x + (Ts/6)*(k1 + 2*k2 + 2*k3 + k4);
end

function dx = f_twotank(x, u, A1, A2, a1, a2, k, g)
    x1 = max(x(1), 0);
    x2 = max(x(2), 0);

    qin  = k*u;
    q12  = a1*sqrt(2*g*x1);
    qout = a2*sqrt(2*g*x2);

    dx1 = (qin - q12)/A1;
    dx2 = (q12 - qout)/A2;

    dx = [dx1; dx2];
end

function [X, Y, idxMap] = build_narx_regressors(y, u, na, nb, nk)
% Build standard SISO NARX regressors
%
% Regression vector at time k:
%   [ y(k-1) ... y(k-na)  u(k-nk) ... u(k-nk-nb+1) ]
%
% Output:
%   Y(row) = y(k)

    y = y(:);
    u = u(:);

    N = length(y);
    maxLag = max(na, nk + nb - 1);

    nRows = N - maxLag;
    X = zeros(nRows, na + nb);
    Y = zeros(nRows, 1);
    idxMap = zeros(nRows, 1);

    row = 0;
    for k = maxLag+1:N
        row = row + 1;

        phi_y = zeros(1, na);
        for i = 1:na
            phi_y(i) = y(k-i);
        end

        phi_u = zeros(1, nb);
        for j = 1:nb
            phi_u(j) = u(k-nk-j+1);
        end

        X(row,:) = [phi_y, phi_u];
        Y(row)   = y(k);
        idxMap(row) = k;
    end
end