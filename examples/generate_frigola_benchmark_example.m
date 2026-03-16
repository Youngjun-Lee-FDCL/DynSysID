function data = generate_frigola_benchmark_example(numExp)
% Generate GP-NARX training/test data for the nonlinear benchmark system
%
% Multi-experiment version:
%   Each experiment is simulated independently.
%   Outputs are stored in cell arrays.
%
% State equation:
%   x(k+1) = a*x(k) + b*x(k)/(1 + x(k)^2) + c*u(k) + v(k)
%   v(k) ~ N(0, q)
%
% Observation equation:
%   y(k) = d*x(k)^2 + e(k)
%   e(k) ~ N(0, r)
%
% Parameters:
%   nominal (a,b,c,d,q,r) = (0.5, 25, 8, 0.05, 0.1, 0.01)

    if nargin < 1 || isempty(numExp)
        numExp = 1;
    end

    rng(1);

    %% Nominal parameters
    a0 = 0.5;
    b0 = 25;
    c0 = 8;
    d0 = 0.05;
    q0 = 0.001;
    r0 = 0.001;

    %% Simulation settings
    N  = 500;
    Ts = 0.1;
    t  = (0:N-1)' * Ts;

    %% NARX settings
    na = 15;
    nb = 15;
    nk = 1;

    %% Preallocate cell arrays
    uCell      = cell(numExp,1);
    yCell      = cell(numExp,1);
    xCell      = cell(numExp,1);
    xNextCell  = cell(numExp,1);
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
        a = a0 * (0.95 + 0.10*rand);
        b = b0 * (0.90 + 0.20*rand);
        c = c0 * (0.90 + 0.20*rand);
        d = d0 * (0.90 + 0.20*rand);

        q = q0 * (0.80 + 0.40*rand);
        r = r0 * (0.80 + 0.40*rand);

        % ------------------------------------------------------------
        % Experiment-specific input
        % ------------------------------------------------------------
        amp1   = 0.9 + 0.3*rand;
        amp2   = 0.2 + 0.2*rand;
        freq1  = 1.2 * (0.90 + 0.20*rand);
        freq2  = 0.35 * (0.80 + 0.40*rand);
        phase2 = 2*pi*rand;

        u = amp1*cos(freq1*(t + 1)) ...
          + amp2*sin(freq2*(t + 1) + phase2) ...
          + 0.05*randn(N,1);

        % ------------------------------------------------------------
        % Simulate latent state and measured output
        % ------------------------------------------------------------
        x = zeros(N+1, 1);
        y = zeros(N, 1);

        % Nonzero random initial condition
        x(1) = randn;

        for k = 1:N
            v = sqrt(q) * randn;
            eMeas = sqrt(r) * randn;

            x(k+1) = a*x(k) + b*x(k)/(1 + x(k)^2) + c*u(k) + v;
            y(k)   = d*x(k)^2 + eMeas;
        end

        % ------------------------------------------------------------
        % Build NARX regressors
        % ------------------------------------------------------------
        [X, Y, idxMap] = build_narx_regressors(y, u, na, nb, nk);

        % ------------------------------------------------------------
        % Train/test split
        % ------------------------------------------------------------
        Ndata = size(X,1);
        Ntr   = round(0.7 * Ndata);

        NtrOriginal = round(0.7 * N);

        % ------------------------------------------------------------
        % Store experiment data
        % ------------------------------------------------------------
        uCell{e}      = u;
        yCell{e}      = y;
        xCell{e}      = x(1:N);
        xNextCell{e}  = x(2:N+1);
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

        expInfo(e).params = struct('a',a,'b',b,'c',c,'d',d,'q',q,'r',r);
        expInfo(e).inputSettings = struct( ...
            'amp1', amp1, ...
            'amp2', amp2, ...
            'freq1', freq1, ...
            'freq2', freq2, ...
            'phase2', phase2);
        expInfo(e).x0 = x(1);
    end

    %% Pack output
    data = struct();
    data.modelName = sprintf('Frigola Nonlinear Benchmark (%d experiments)', numExp);
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
    data.xNextCell  = xNextCell;
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

    data.trueParams = struct('a',a0,'b',b0,'c',c0,'d',d0,'q',q0,'r',r0);
    data.expInfo = expInfo;

    %% Backward compatibility for single experiment
    if numExp == 1
        data.u = uCell{1};
        data.y = yCell{1};
        data.x = xCell{1};
        data.x_next = xNextCell{1};

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