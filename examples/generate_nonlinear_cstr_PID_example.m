function data = generate_nonlinear_cstr_PID_example(numExp)
% Generate data for a nonlinear 4-input 3-output CSTR-like system
% using PID controllers for step-reference tracking.
%
% Multi-experiment version:
%   Each experiment is simulated independently.
%   Outputs are stored in cell arrays.
%
% Input
%   numExp : number of independent experiments
%
% Output
%   data struct containing multi-experiment datasets and metadata

    if nargin < 1 || isempty(numExp)
        numExp = 1;
    end

    rng(1);

    %% Simulation settings
    Ts = 0.05;
    N  = 1200;
    t  = (0:N-1)' * Ts;

    %% Physical / model parameters
    V         = 1.0;
    k0        = 1.2e3;
    E_over_R  = 4000;   % reduced activation term
    alpha     = 8.0;    % temperature rise coefficient
    beta      = 0.08;   % heat input coefficient

    %% Nominal operating point / input bias
    uBias = [1.0, 1.0, 350.0, 0.0];  % [F, CAf, Tf, Q]

    %% Noise settings
    sigmaW = 0.01*[0.002, 0.10, 0.002];   % process noise std for states
    sigmaV = 0.01*[0.005, 0.20, 0.005];   % measurement noise std for outputs

    %% Nominal PID gains for 3 controlled outputs
    % y1 -> u1 mainly
    % y2 -> u4 mainly
    % y3 -> u2 mainly
    % u3 is used as an auxiliary coupled input
    Kp0 = [1.2;  0.8;  1.0];
    Ki0 = [0.08; 0.04; 0.06];
    Kd0 = [0.02; 0.01; 0.02];

    %% Input bounds
    uMin = [0.2, 0.2, 300.0, -20.0];
    uMax = [2.0, 2.0, 420.0,  20.0];

    %% Dimensions
    nx = 3;
    ny = 3;
    nu = 4;

    %% Suggested model orders for identification
    na = 4 * ones(3,3);
    nb = 4 * ones(3,4);
    nk = 1 * ones(3,4);

    %% Train/validation split
    Ntr = round(0.7 * N);

    %% Preallocate cell arrays
    xCell    = cell(numExp,1);
    yCell    = cell(numExp,1);
    uCell    = cell(numExp,1);
    rCell    = cell(numExp,1);
    tCell    = cell(numExp,1);

    xEstCell = cell(numExp,1);
    yEstCell = cell(numExp,1);
    uEstCell = cell(numExp,1);
    rEstCell = cell(numExp,1);

    xValCell = cell(numExp,1);
    yValCell = cell(numExp,1);
    uValCell = cell(numExp,1);
    rValCell = cell(numExp,1);

    idxValCell = cell(numExp,1);

    expInfo = repmat(struct(), numExp, 1);

    %% Run multiple independent experiments
    for e = 1:numExp
        % Slight variation in PID gains
        gainScaleP = 0.85 + 0.30*rand;
        gainScaleI = 0.85 + 0.30*rand;
        gainScaleD = 0.85 + 0.30*rand;

        Kp = gainScaleP * Kp0;
        Ki = gainScaleI * Ki0;
        Kd = gainScaleD * Kd0;

        % Slight variation in initial state
        x0 = [ ...
            max(0.0, 0.8  + 0.05*randn), ...
            max(250, 350.0 + 3.0*randn), ...
            max(0.0, 0.15 + 0.03*randn)];

        % Experiment-specific references
        r = generate_reference_profile_cstr(t, e);

        % Simulate one experiment
        [x, y, u] = simulate_one_cstr_experiment( ...
            t, r, x0, ...
            V, k0, E_over_R, alpha, beta, ...
            Kp, Ki, Kd, ...
            uBias, uMin, uMax, ...
            sigmaW, sigmaV);

        % Store full data
        xCell{e} = x;
        yCell{e} = y;
        uCell{e} = u;
        rCell{e} = r;
        tCell{e} = t;

        % Store split data
        xEstCell{e} = x(1:Ntr,:);
        yEstCell{e} = y(1:Ntr,:);
        uEstCell{e} = u(1:Ntr,:);
        rEstCell{e} = r(1:Ntr,:);

        xValCell{e} = x(Ntr+1:end,:);
        yValCell{e} = y(Ntr+1:end,:);
        uValCell{e} = u(Ntr+1:end,:);
        rValCell{e} = r(Ntr+1:end,:);

        idxValCell{e} = (Ntr+1:N).';

        % Per-experiment info
        expInfo(e).Kp = Kp;
        expInfo(e).Ki = Ki;
        expInfo(e).Kd = Kd;
        expInfo(e).x0 = x0;
    end

    %% Output struct
    data = struct();
    data.modelName = sprintf('Nonlinear CSTR 4x3 PID Tracking Example (%d experiments)', numExp);
    data.isMultiExperiment = (numExp > 1);
    data.numExp = numExp;

    data.Ts = Ts;
    data.N  = N;
    data.t  = t;    % common time axis

    data.nx = nx;
    data.ny = ny;
    data.nu = nu;

    data.uBias = uBias;
    data.uMin  = uMin;
    data.uMax  = uMax;

    data.params = struct( ...
        'V', V, ...
        'k0', k0, ...
        'E_over_R', E_over_R, ...
        'alpha', alpha, ...
        'beta', beta);

    data.sigmaW = sigmaW;
    data.sigmaV = sigmaV;

    data.na = na;
    data.nb = nb;
    data.nk = nk;

    % Multi-experiment cell data
    data.xCell = xCell;
    data.yCell = yCell;
    data.uCell = uCell;
    data.rCell = rCell;
    data.tCell = tCell;

    data.xEstCell = xEstCell;
    data.yEstCell = yEstCell;
    data.uEstCell = uEstCell;
    data.rEstCell = rEstCell;

    data.xValCell = xValCell;
    data.yValCell = yValCell;
    data.uValCell = uValCell;
    data.rValCell = rValCell;

    data.idxValCell = idxValCell;
    data.expInfo = expInfo;

    % Backward compatibility for single experiment
    if numExp == 1
        data.x = xCell{1};
        data.y = yCell{1};
        data.u = uCell{1};
        data.r = rCell{1};

        data.xEst = xEstCell{1};
        data.yEst = yEstCell{1};
        data.uEst = uEstCell{1};
        data.rEst = rEstCell{1};

        data.xVal = xValCell{1};
        data.yVal = yValCell{1};
        data.uVal = uValCell{1};
        data.rVal = rValCell{1};

        data.idxVal = idxValCell{1};

        data.Kp = expInfo(1).Kp;
        data.Ki = expInfo(1).Ki;
        data.Kd = expInfo(1).Kd;
    end
end

function [x, y, u] = simulate_one_cstr_experiment( ...
    t, r, x0, ...
    V, k0, E_over_R, alpha, beta, ...
    Kp, Ki, Kd, ...
    uBias, uMin, uMax, ...
    sigmaW, sigmaV)
% Simulate one closed-loop CSTR experiment

    Ts = t(2) - t(1);
    N  = numel(t);

    nx = 3;
    ny = 3;
    nu = 4;

    x = zeros(N, nx);
    y = zeros(N, ny);
    u = zeros(N, nu);

    eInt  = zeros(1,ny);
    ePrev = zeros(1,ny);

    x(1,:) = x0;
    u(1,:) = uBias;
    y(1,:) = x(1,:) + sigmaV .* randn(1,ny);

    for k = 1:N-1
        % Tracking error
        e = r(k,:) - y(k,:);

        % PID terms
        eInt = eInt + e * Ts;
        eDer = (e - ePrev) / Ts;

        pidOut = (Kp .* e.' + Ki .* eInt.' + Kd .* eDer.').';

        % Input mapping:
        % u1 <- y1 loop
        % u2 <- y3 loop
        % u3 <- weak coupling from y1,y2,y3 loops
        % u4 <- y2 loop
        uRaw = uBias + [ ...
            pidOut(1), ...
            pidOut(3), ...
            8.0*pidOut(2) + 4.0*pidOut(1) - 3.0*pidOut(3), ...
            6.0*pidOut(2)];

        % Saturation
        u(k,:) = min(max(uRaw, uMin), uMax);

        % Simple anti-windup
        satMask = abs(u(k,:) - uRaw) > 1e-12;
        if satMask(1), eInt(1) = eInt(1) - e(1)*Ts; end
        if satMask(2), eInt(3) = eInt(3) - e(3)*Ts; end
        if satMask(4), eInt(2) = eInt(2) - e(2)*Ts; end

        % Current states and inputs
        CA  = x(k,1);
        T   = x(k,2);
        CB  = x(k,3);

        F   = u(k,1);
        CAf = u(k,2);
        Tf  = u(k,3);
        Q   = u(k,4);

        % Nonlinear reaction rate
        kT = k0 * exp(-E_over_R / max(T,250));

        % Continuous-time dynamics
        dCA = F/V*(CAf - CA) - kT*CA;
        dT  = F/V*(Tf - T) + alpha*kT*CA + beta*Q;
        dCB = -F/V*CB + kT*CA;

        % Euler integration + process noise
        w = sigmaW .* randn(1,nx);

        x(k+1,1) = CA + Ts*dCA + w(1);
        x(k+1,2) = T  + Ts*dT  + w(2);
        x(k+1,3) = CB + Ts*dCB + w(3);

        % Prevent unphysical states
        x(k+1,1) = max(x(k+1,1), 0.0);
        x(k+1,2) = max(x(k+1,2), 250.0);
        x(k+1,3) = max(x(k+1,3), 0.0);

        % Output with measurement noise
        v = sigmaV .* randn(1,ny);
        y(k+1,:) = x(k+1,:) + v;

        ePrev = e;
    end

    % Final input sample
    u(N,:) = u(N-1,:);
end

function r = generate_reference_profile_cstr(t, expIdx)
% Generate experiment-specific reference profile for CSTR

    N = numel(t);
    r = zeros(N,3);

    baseTimes = [0 10 20 30 40 50];

    % Small switching-time jitter
    jitter = [0 cumsum(0.25*randn(1, numel(baseTimes)-1))];
    changeTimes = baseTimes + jitter;
    changeTimes(1) = 0;
    changeTimes = max(changeTimes, 0);
    changeTimes = sort(changeTimes);

    % Base reference levels
    lev1 = [0.80 -0.65 -0.90 -0.70 0.85 0.75];
    lev2 = [350 360 345 370 355 365];
    lev3 = [0.15 -0.25 -0.10 -0.30 0.18 0.22];

    % Experiment-dependent amplitude scaling
    ampScale1 = 0.85 + 0.30*rand;
    ampScale2 = 0.95 + 0.10*rand;
    ampScale3 = 0.85 + 0.30*rand;

    lev1 = ampScale1 * lev1 + 0.02*randn(size(lev1));
    lev2 = 350 + ampScale2 * (lev2 - 350) + 0.8*randn(size(lev2));
    lev3 = ampScale3 * lev3 + 0.01*randn(size(lev3));

    % Small experiment-dependent biases
    lev1 = lev1 + 0.01*sin(0.6*expIdx);
    lev2 = lev2 + 0.8*cos(0.4*expIdx);
    lev3 = lev3 + 0.01*sin(0.3*expIdx);

    r(:,1) = make_step_profile(t, changeTimes, lev1);
    r(:,2) = make_step_profile(t, changeTimes, lev2);
    r(:,3) = make_step_profile(t, changeTimes, lev3);
end

function ref = make_step_profile(t, changeTimes, levels)
% Build a piecewise-constant step reference profile.

    ref = zeros(size(t));
    for i = 1:length(changeTimes)
        idx = t >= changeTimes(i);
        ref(idx) = levels(i);
    end
end