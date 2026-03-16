function data = generate_nonlinear_aircraft43_PID_example(numExp)
% Generate data for a nonlinear 4-input 3-output aircraft-like system
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

    %% Common simulation settings
    Ts = 0.05;
    N  = 1500;
    t  = (0:N-1)' * Ts;

    %% Noise settings
    sigmaW = 0.01*[0.003 0.003 0.003 0.002 0.002 0.010];
    sigmaV = 0.01*[0.005 0.005 0.005];

    %% Nominal PID gains
    Kp0 = [2.8; 2.5; 2.8];
    Ki0 = [0.25; 0.20; 0.12];
    Kd0 = [0.12; 0.10; 0.08];

    %% Input limits
    uMin = [-0.4, -0.4, -0.4, 0.1];
    uMax = [ 0.4,  0.4,  0.4, 1.0];

    %% Input bias
    uBias = [0.0, 0.0, 0.0, 0.5];

    %% Dimensions
    nx = 6;
    ny = 3;
    nu = 4;

    %% Suggested model orders
    na = 4 * ones(3,3);
    nb = 4 * ones(3,4);
    nk = 1 * ones(3,4);

    %% Train/validation split ratio
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
        % Slightly vary PID gains per experiment
        gainScaleP = 0.85 + 0.30*rand;
        gainScaleI = 0.85 + 0.30*rand;
        gainScaleD = 0.85 + 0.30*rand;

        Kp = gainScaleP * Kp0;
        Ki = gainScaleI * Ki0;
        Kd = gainScaleD * Kd0;

        % Slightly vary initial condition
        x0 = [ ...
            0.03*randn, ...           % q
            0.03*randn, ...           % p
            0.03*randn, ...           % r
            0.05*randn, ...           % theta
            0.05*randn, ...           % phi
            100 + 2.0*randn];         % V

        % Generate experiment-specific reference
        r = generate_reference_profile(t, e);

        % Simulate one experiment
        [x, y, u] = simulate_one_experiment( ...
            t, r, x0, ...
            Kp, Ki, Kd, ...
            sigmaW, sigmaV, ...
            uMin, uMax, uBias);

        % Store full trajectories
        xCell{e} = x;
        yCell{e} = y;
        uCell{e} = u;
        rCell{e} = r;
        tCell{e} = t;

        % Store split trajectories
        xEstCell{e} = x(1:Ntr,:);
        yEstCell{e} = y(1:Ntr,:);
        uEstCell{e} = u(1:Ntr,:);
        rEstCell{e} = r(1:Ntr,:);

        xValCell{e} = x(Ntr+1:end,:);
        yValCell{e} = y(Ntr+1:end,:);
        uValCell{e} = u(Ntr+1:end,:);
        rValCell{e} = r(Ntr+1:end,:);

        idxValCell{e} = (Ntr+1:N).';

        % Per-experiment metadata
        expInfo(e).Kp = Kp;
        expInfo(e).Ki = Ki;
        expInfo(e).Kd = Kd;
        expInfo(e).x0 = x0;
    end

    %% Output struct
    data = struct();
    data.modelName = sprintf('Nonlinear Aircraft-like 4x3 PID Tracking Example (%d experiments)', numExp);
    data.isMultiExperiment = (numExp > 1);
    data.numExp = numExp;

    data.Ts = Ts;
    data.N  = N;
    data.t  = t;   % common time axis

    data.nx = nx;
    data.ny = ny;
    data.nu = nu;

    data.sigmaW = sigmaW;
    data.sigmaV = sigmaV;

    data.uBias = uBias;
    data.uMin  = uMin;
    data.uMax  = uMax;

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

function [x, y, u] = simulate_one_experiment(t, r, x0, Kp, Ki, Kd, sigmaW, sigmaV, uMin, uMax, uBias)
% Simulate one closed-loop experiment

    Ts = t(2) - t(1);
    N  = numel(t);

    nx = 6;
    ny = 3;
    nu = 4;

    x = zeros(N, nx);
    y = zeros(N, ny);
    u = zeros(N, nu);

    eInt  = zeros(1,ny);
    ePrev = zeros(1,ny);

    x(1,:) = x0;
    u(1,:) = uBias;
    y(1,:) = [x(1,4), x(1,5), x(1,3)] + sigmaV .* randn(1,ny);

    for k = 1:N-1
        yk = y(k,:);

        % Tracking error
        e = r(k,:) - yk;

        % PID terms
        eInt = eInt + e * Ts;
        eDer = (e - ePrev) / Ts;

        pidOut = (Kp .* e.' + Ki .* eInt.' + Kd .* eDer.').';

        % Input mapping
        uRaw = uBias + [ ...
            pidOut(1), ...
            pidOut(2), ...
            pidOut(3), ...
            0.15*pidOut(1) - 0.10*pidOut(2)];

        % Saturation
        u(k,:) = min(max(uRaw, uMin), uMax);

        % Anti-windup
        satMask = abs(u(k,:) - uRaw) > 1e-12;
        if satMask(1), eInt(1) = eInt(1) - e(1)*Ts; end
        if satMask(2), eInt(2) = eInt(2) - e(2)*Ts; end
        if satMask(3), eInt(3) = eInt(3) - e(3)*Ts; end

        % State extraction
        q     = x(k,1);
        p     = x(k,2);
        rYaw  = x(k,3);
        theta = x(k,4);
        phi   = x(k,5);
        V     = x(k,6);

        de = u(k,1);
        da = u(k,2);
        dr = u(k,3);
        dt = u(k,4);

        % Nonlinear aircraft-like dynamics
        qDot = -1.10*q + 0.15*p*rYaw - 0.08*sin(theta) ...
               + 2.50*de + 0.30*dt;

        pDot = -1.00*p - 0.12*q*rYaw - 0.06*sin(phi) ...
               + 2.20*da + 0.05*dr;

        rDot = -0.85*rYaw + 0.10*p*q + 0.04*sin(phi)*V/100 ...
               + 1.60*dr + 0.08*da;

        thetaDot = q - 0.25*theta + 0.04*sin(phi);
        phiDot   = p - 0.22*phi + 0.03*sin(theta);

        VDot = -0.06*(V - 100) + 8.0*(dt - 0.5) ...
               - 1.2*abs(de) - 0.8*abs(da) - 0.5*abs(dr);

        % Euler update + process noise
        w = sigmaW .* randn(1,nx);

        x(k+1,1) = q     + Ts*qDot     + w(1);
        x(k+1,2) = p     + Ts*pDot     + w(2);
        x(k+1,3) = rYaw  + Ts*rDot     + w(3);
        x(k+1,4) = theta + Ts*thetaDot + w(4);
        x(k+1,5) = phi   + Ts*phiDot   + w(5);
        x(k+1,6) = V     + Ts*VDot     + w(6);

        % Optional clipping
        x(k+1,4) = min(max(x(k+1,4), -0.7), 0.7);
        x(k+1,5) = min(max(x(k+1,5), -0.8), 0.8);
        x(k+1,6) = max(x(k+1,6), 40);

        % Output equation
        v = sigmaV .* randn(1,ny);
        y(k+1,:) = [x(k+1,4), x(k+1,5), x(k+1,3)] + v;

        ePrev = e;
    end

    u(N,:) = u(N-1,:);
end

function r = generate_reference_profile(t, expIdx)
% Experiment-specific reference profile

    N = numel(t);
    r = zeros(N,3);

    baseTimes = [0 12 24 36 48 60 72 84 96];

    % Small random perturbation of switching times
    jitter = [0 cumsum(0.3*randn(1, numel(baseTimes)-1))];
    changeTimes = baseTimes + jitter;
    changeTimes(1) = 0;
    changeTimes = max(changeTimes, 0);
    changeTimes = sort(changeTimes);

    % Base levels + experiment-dependent perturbation
    lev1 = [ 0.00  0.08 -0.06  0.10 -0.08  0.05  0.10 -0.08 -0.06];
    lev2 = [ 0.00 -0.10  0.08 -0.12  0.10 -0.06  0.10 -0.08 -0.06];
    lev3 = [ 0.00  0.05 -0.10  0.06 -0.05  0.03 -0.10  0.08 -0.06];

    ampScale = 0.85 + 0.30*rand;

    lev1 = ampScale * lev1 + 0.01*randn(size(lev1));
    lev2 = ampScale * lev2 + 0.01*randn(size(lev2));
    lev3 = ampScale * lev3 + 0.01*randn(size(lev3));

    % Slight experiment-dependent bias
    lev1 = lev1 + 0.01*sin(0.7*expIdx);
    lev2 = lev2 + 0.01*cos(0.5*expIdx);
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