function data = generate_nonlinear_twotank_example()
% Generate GP-NARX training/test data for nonlinear two-tank system
%
% Continuous-time model:
%   dx1/dt = (1/A1) * (k*u - a1*sqrt(2*g*x1))
%   dx2/dt = (1/A2) * (a1*sqrt(2*g*x1) - a2*sqrt(2*g*x2))
%   y      = x2 + noise
%
% State:
%   x1 : upper tank level
%   x2 : lower tank level

    rng(1);

    %% Simulation settings
    Ts = 0.2;
    N  = 1000;
    t  = (0:N-1)' * Ts;

    %% Input signal (pump voltage / flow command)
    u = 3.0 ...
      + 0.8*sin(2*pi*0.003*t) ...
      + 0.5*sin(2*pi*0.011*t + 0.7) ...
      + 0.2*randn(N,1);

    % Pump input should be nonnegative
    u = max(u, 0);

    %% True parameters
    A1 = 0.5;
    A2 = 0.25;
    a1 = 0.02;
    a2 = 0.015;
    k  = 0.005;
    g  = 9.81;

    sigmaY = 0.0001;

    %% Simulate
    x = zeros(2, N);   % x(1,:) = upper tank level, x(2,:) = lower tank level
    y = zeros(N, 1);

    % Optional nonzero initial condition
    x(:,1) = [0.2; 0.1];

    for kk = 1:N-1
        x(:,kk+1) = rk4_step_twotank(x(:,kk), u(kk), Ts, A1, A2, a1, a2, k, g);

        % Enforce nonnegative water levels
        x(:,kk+1) = max(x(:,kk+1), 0);

        y(kk) = x(2,kk) + sigmaY*randn;
    end
    y(N) = x(2,N) + sigmaY*randn;

    %% NARX settings
    na = 10;
    nb = 10;
    nk = 1;

    [X, Y, idxMap] = build_narx_regressors(y, u, na, nb, nk);

    %% Split
    Ndata = size(X,1);
    Ntr   = round(0.7*Ndata);

    %% Pack data
    data = struct();
    data.modelName = 'Nonlinear Two-Tank';
    data.Ts = Ts;
    data.t  = t;
    data.u  = u;
    data.y  = y;
    data.x  = x;

    data.na = na;
    data.nb = nb;
    data.nk = nk;

    data.X = X;
    data.Y = Y;
    data.Xtr = X(1:Ntr,:);
    data.Ytr = Y(1:Ntr);
    data.Xte = X(Ntr+1:end,:);
    data.Yte = Y(Ntr+1:end);
    data.idxTe = idxMap(Ntr+1:end);

    %% Metadata
    data.params = struct( ...
        'A1', A1, ...
        'A2', A2, ...
        'a1', a1, ...
        'a2', a2, ...
        'k',  k, ...
        'g',  g, ...
        'sigmaY', sigmaY);
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