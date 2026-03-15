function data = generate_linear_mimo3_PID_example()
% Generate data for a 3x3 linear MIMO discrete-time system
% using PID controllers for step-reference tracking.
%
% State-space model:
%   x(k+1) = A x(k) + B u(k) + w(k)
%   y(k)   = C x(k) + D u(k) + v(k)
%
% Inputs : 3
% Outputs: 3
%
% Output:
%   data struct containing train/validation split and metadata

    rng(1);

    %% Simulation settings
    Ts = 0.1;
    N  = 1000;
    t  = (0:N-1)' * Ts;

    %% System matrices
    % 6-state stable discrete-time system
    A = [ 0.82  0.10  0.00  0.00  0.00  0.00;
         -0.18  0.75  0.12  0.00  0.00  0.00;
          0.00 -0.10  0.78  0.08  0.00  0.00;
          0.00  0.00 -0.15  0.72  0.10  0.00;
          0.00  0.00  0.00 -0.12  0.76  0.10;
          0.00  0.00  0.00  0.00 -0.20  0.70];

    B = [0.15  0.00  0.05;
         0.10  0.08  0.00;
         0.00  0.12  0.06;
         0.05  0.00  0.10;
         0.00  0.07  0.12;
         0.08  0.04  0.00];

    C = [1.0  0.0  0.2  0.0  0.0  0.0;
         0.0  1.0  0.0  0.3  0.0  0.0;
         0.1  0.0  0.0  0.0  1.0  0.2];

    D = zeros(3,3);

    %% Noise settings
    sigmaW = 0.01;   % process noise std
    sigmaV = 0.02;   % measurement noise std

    %% PID gains
    % Decentralized PID: each input channel is assigned to each output channel
    Kp = [1.8; 1.6; 1.7];
    Ki = [0.25; 0.20; 0.22];
    Kd = [0.08; 0.07; 0.09];

    uMin = -5;
    uMax =  5;

    %% Step reference generation
    % Piecewise-constant multi-step references for each output
    r = zeros(N,3);

    r(:,1) = make_step_profile(t, [0 15 30 45 60 75 90], [0.0  1.0 -1.0  1.0  -1.0, 1.0, -1.0]);
    r(:,2) = make_step_profile(t, [0 15 30 45 60 75 90], [0.0  1.0 -1.0  1.0  -1.0, 1.0, -1.0]);
    r(:,3) = make_step_profile(t, [0 15 30 45 60 75 90], [0.0  1.0 -1.0  1.0  -1.0, 1.0, -1.0]);

    %% Preallocation
    nx = size(A,1);
    ny = size(C,1);
    nu = size(B,2);

    x = zeros(N, nx);
    y = zeros(N, ny);
    u = zeros(N, nu);

    eInt  = zeros(1,ny);
    ePrev = zeros(1,ny);

    %% Initial output
    y(1,:) = (C*x(1,:).' + D*u(1,:).' + sigmaV*randn(ny,1)).';

    %% Closed-loop simulation with PID
    for k = 1:N-1
        % Tracking error
        e = r(k,:) - y(k,:);

        % Integral and derivative terms
        eInt = eInt + e * Ts;
        eDer = (e - ePrev) / Ts;

        % PID law (decentralized)
        uRaw = Kp .* e.' + Ki .* eInt.' + Kd .* eDer.';
        u(k,:) = uRaw.';

        % Saturation
        u(k,:) = min(max(u(k,:), uMin), uMax);

        % Simple anti-windup: freeze integrator if saturated
        satMask = (u(k,:) ~= uRaw.');
        eInt(satMask) = eInt(satMask) - e(satMask) * Ts;

        % Plant update
        w = sigmaW * randn(nx,1);
        v = sigmaV * randn(ny,1);

        x(k+1,:) = (A*x(k,:).' + B*u(k,:).' + w).';
        y(k+1,:) = (C*x(k+1,:).' + D*u(k,:).' + v).';

        % Save previous error
        ePrev = e;
    end

    % Fill last input sample
    u(N,:) = u(N-1,:);

    %% Suggested linear model orders
    na = [2 2 2;
          2 2 2;
          2 2 2];

    nb = [2 2 2;
          2 2 2;
          2 2 2];

    nk = [1 1 1;
          1 1 1;
          1 1 1];

    %% Train/validation split
    Ntr = round(0.7 * N);

    data = struct();
    data.modelName = '3x3 Linear MIMO PID Tracking Example';
    data.Ts = Ts;
    data.t  = t;

    data.A = A;
    data.B = B;
    data.C = C;
    data.D = D;

    data.x = x;
    data.r = r;   % reference
    data.u = u;   % input, N x 3
    data.y = y;   % output, N x 3

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
%
% Inputs:
%   t           : time vector (N x 1)
%   changeTimes : times when reference changes
%   levels      : reference levels for each interval
%
% Example:
%   changeTimes = [0 10 20]
%   levels      = [0 1 -0.5]

    ref = zeros(size(t));

    for i = 1:length(changeTimes)
        idx = t >= changeTimes(i);
        ref(idx) = levels(i);
    end
end