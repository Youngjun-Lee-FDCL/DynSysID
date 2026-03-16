function data = generate_nonlinear_cstr_PID_example()
% Generate data for a nonlinear 4-input 3-output CSTR-like system
% using PID controllers for step-reference tracking.
%
% States:
%   x1 = CA   : reactant concentration
%   x2 = T    : reactor temperature
%   x3 = CB   : product concentration
%
% Inputs:
%   u1 = F    : inlet flow rate
%   u2 = CAf  : feed concentration
%   u3 = Tf   : feed temperature
%   u4 = Q    : heating/cooling term
%
% Outputs:
%   y1 = CA
%   y2 = T
%   y3 = CB
%
% Continuous-time nonlinear model:
%   dCA/dt = F/V*(CAf - CA) - k(T)*CA
%   dT/dt  = F/V*(Tf - T) + alpha*k(T)*CA + beta*Q
%   dCB/dt = -F/V*CB + k(T)*CA
%
% where
%   k(T) = k0 * exp(-E_over_R / T)
%
% Output:
%   data struct containing train/validation split and metadata

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

    %% PID gains for 3 controlled outputs
    % y1 -> u1 mainly
    % y2 -> u4 mainly
    % y3 -> u2 mainly
    %
    % u3 is used as an auxiliary coupled input
    Kp = [1.2;  0.8;  1.0];
    Ki = [0.08; 0.04; 0.06];
    Kd = [0.02; 0.01; 0.02];

    %% Input bounds
    uMin = [0.2, 0.2, 300.0, -20.0];
    uMax = [2.0, 2.0, 420.0,  20.0];

    %% Step references for outputs [CA, T, CB]
    r = zeros(N,3);
    changeTimes = [0 10 20 30 40 50];
    r(:,1) = make_step_profile(t, changeTimes, [0.80 -0.65 -0.90 -0.70 0.85 0.75]);
    r(:,2) = make_step_profile(t, changeTimes, [350 360 345 370 355 365]);
    r(:,3) = make_step_profile(t, changeTimes, [0.15 -0.25 -0.10 -0.30 0.18 0.22]);

    %% Dimensions
    nx = 3;
    ny = 3;
    nu = 4;

    %% Preallocation
    x = zeros(N, nx);
    y = zeros(N, ny);
    u = zeros(N, nu);

    eInt  = zeros(1,ny);
    ePrev = zeros(1,ny);

    %% Initial state near nominal
    x(1,:) = [0.8, 350.0, 0.15];
    u(1,:) = uBias;

    % Initial output
    y(1,:) = x(1,:) + sigmaV .* randn(1,ny);

    %% Closed-loop simulation
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

        % map actuator saturation back to corresponding integrators
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

    %% Suggested model orders for identification
    na = 4 * ones(3,3);
    nb = 4 * ones(3,4);
    nk = 1 * ones(3,4);

    %% Train/validation split
    Ntr = round(0.7 * N);

    data = struct();
    data.modelName = 'Nonlinear CSTR 4x3 PID Tracking Example';
    data.Ts = Ts;
    data.t  = t;

    data.x = x;
    data.r = r;
    data.u = u;
    data.y = y;

    data.uBias = uBias;
    data.params = struct('V',V,'k0',k0,'E_over_R',E_over_R,'alpha',alpha,'beta',beta);

    data.Kp = Kp;
    data.Ki = Ki;
    data.Kd = Kd;

    data.na = na;
    data.nb = nb;
    data.nk = nk;

    data.uEst = u(1:Ntr,:);
    data.yEst = y(1:Ntr,:);
    data.rEst = r(1:Ntr,:);

    data.uVal = u(Ntr+1:end,:);
    data.yVal = y(Ntr+1:end,:);
    data.rVal = r(Ntr+1:end,:);

    data.idxVal = (Ntr+1:N).';
end

function ref = make_step_profile(t, changeTimes, levels)
% Build a piecewise-constant step reference profile.

    ref = zeros(size(t));
    for i = 1:length(changeTimes)
        idx = t >= changeTimes(i);
        ref(idx) = levels(i);
    end
end