function data = generate_nonlinear_msd_example(numExp)
% Generate GP-NARX training/test data for nonlinear mass-spring-damper system
%
% Multi-experiment version:
%   Each experiment is simulated independently.
%   Outputs are stored in cell arrays.
%
% State:
%   x1 = position
%   x2 = velocity
%
% Dynamics:
%   m*xdd + c*xd + c3*xd^3 + k*x + k3*x^3 = u
%
% Output:
%   y = x1 + noise

    if nargin < 1 || isempty(numExp)
        numExp = 1;
    end

    rng(1);

    %% Simulation settings
    Ts = 0.02;
    N  = 800;
    t  = (0:N-1)' * Ts;

    %% Nominal parameters
    m0  = 1.0;
    c0  = 0.8;
    k0  = 20.0;
    c30 = 0.4;
    k30 = 15.0;

    sigmaY0 = 0.003;

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
        % Slightly perturbed parameters per experiment
        % ------------------------------------------------------------
        m  = m0  * (0.95 + 0.10*rand);
        c  = c0  * (0.85 + 0.30*rand);
        k  = k0  * (0.90 + 0.20*rand);
        c3 = c30 * (0.80 + 0.40*rand);
        k3 = k30 * (0.80 + 0.40*rand);

        sigmaY = sigmaY0 * (0.8 + 0.4*rand);

        % ------------------------------------------------------------
        % Experiment-specific input
        % ------------------------------------------------------------
        sigmaU = 0.35 + 0.25*rand;
        amp1   = 0.8 + 0.4*rand;
        amp2   = 0.5 + 0.3*rand;
        freq1  = 0.4 * (0.85 + 0.30*rand);
        freq2  = 1.1 * (0.85 + 0.30*rand);
        phase2 = 2*pi*rand;

        u = amp1*sin(2*pi*freq1*t) ...
          + amp2*sin(2*pi*freq2*t + phase2) ...
          + sigmaU*randn(N,1);

        % Add a few random step-like perturbations
        nSteps = 2 + randi(2);
        stepTimes = sort(randsample((50:N-100), nSteps));
        for kk = 1:nSteps
            u(stepTimes(kk):end) = u(stepTimes(kk):end) + 0.3*randn;
        end

        % ------------------------------------------------------------
        % Simulate
        % ------------------------------------------------------------
        x = zeros(2,N);
        y = zeros(N,1);

        % Nonzero initial condition
        x(:,1) = [0.05*randn; 0.05*randn];

        for kk = 1:N-1
            x(:,kk+1) = rk4_step_nonlinear(x(:,kk), u(kk), Ts, m, c, k, c3, k3);
            y(kk) = x(1,kk) + sigmaY*randn;
        end
        y(N) = x(1,N) + sigmaY*randn;

        % ------------------------------------------------------------
        % Build NARX regressors
        % ------------------------------------------------------------
        [X, Y, idxMap] = build_narx_regressors(y, u, na, nb, nk);

        % ------------------------------------------------------------
        % Split
        % ------------------------------------------------------------
        Ndata = size(X,1);
        Ntr   = round(0.7*Ndata);

        NtrOriginal = round(0.7*N);

        % ------------------------------------------------------------
        % Store experiment data
        % ------------------------------------------------------------
        uCell{e}      = u;
        yCell{e}      = y;
        xCell{e}      = x;
        tCell{e}      = t;

        XCell{e}      = X;
        YCell{e}      = Y;
        idxMapCell{e} = idxMap;

        XtrCell{e}    = X(1:Ntr,:);
        YtrCell{e}    = Y(1:Ntr);
        XteCell{e}    = X(Ntr+1:end,:);
        YteCell{e}    = Y(Ntr+1:end);
        idxTeCell{e}  = idxMap(Ntr+1:end);

        uEstCell{e}   = u(1:NtrOriginal,:);
        yEstCell{e}   = y(1:NtrOriginal,:);
        uValCell{e}   = u(NtrOriginal+1:end,:);
        yValCell{e}   = y(NtrOriginal+1:end,:);
        idxValCell{e} = (NtrOriginal+1:N).';

        expInfo(e).params = struct( ...
            'm', m, ...
            'c', c, ...
            'k', k, ...
            'c3', c3, ...
            'k3', k3, ...
            'sigmaY', sigmaY);

        expInfo(e).inputSettings = struct( ...
            'sigmaU', sigmaU, ...
            'amp1', amp1, ...
            'amp2', amp2, ...
            'freq1', freq1, ...
            'freq2', freq2, ...
            'phase2', phase2);

        expInfo(e).x0 = x(:,1);
    end

    %% Pack output
    data = struct();
    data.modelName = sprintf('Nonlinear MSD (%d experiments)', numExp);
    data.isMultiExperiment = (numExp > 1);
    data.numExp = numExp;

    data.Ts = Ts;
    data.t  = t;
    data.N  = N;

    data.na = na;
    data.nb = nb;
    data.nk = nk;

    data.uCell      = uCell;
    data.yCell      = yCell;
    data.xCell      = xCell;
    data.tCell      = tCell;

    data.XCell      = XCell;
    data.YCell      = YCell;
    data.idxMapCell = idxMapCell;

    data.XtrCell    = XtrCell;
    data.YtrCell    = YtrCell;
    data.XteCell    = XteCell;
    data.YteCell    = YteCell;
    data.idxTeCell  = idxTeCell;

    data.uEstCell   = uEstCell;
    data.yEstCell   = yEstCell;
    data.uValCell   = uValCell;
    data.yValCell   = yValCell;
    data.idxValCell = idxValCell;

    data.nominalParams = struct( ...
        'm', m0, ...
        'c', c0, ...
        'k', k0, ...
        'c3', c30, ...
        'k3', k30, ...
        'sigmaY', sigmaY0);

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

function xNext = rk4_step_nonlinear(x, u, Ts, m, c, k, c3, k3)
    k_1 = f_nonlinear(x, u, m, c, k, c3, k3);
    k_2 = f_nonlinear(x + 0.5*Ts*k_1, u, m, c, k, c3, k3);
    k_3 = f_nonlinear(x + 0.5*Ts*k_2, u, m, c, k, c3, k3);
    k_4 = f_nonlinear(x + Ts*k_3,     u, m, c, k, c3, k3);

    xNext = x + (Ts/6)*(k_1 + 2*k_2 + 2*k_3 + k_4);
end

function dx = f_nonlinear(x, u, m, c, k, c3, k3)
    pos = x(1);
    vel = x(2);
    acc = (u - c*vel - c3*vel^3 - k*pos - k3*pos^3)/m;
    dx = [vel; acc];
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