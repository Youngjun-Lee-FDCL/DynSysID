function data = generate_linear_mimo3_example()
% Generate data for a 3x3 linear MIMO discrete-time system
%
% State-space model:
%   x(k+1) = A x(k) + B u(k) + w(k)
%   y(k)   = C x(k) + D u(k) + v(k)
%
% Inputs : 3
% Outputs: 3
%
% Output:
%   data struct containing train/test split and metadata

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

    D = [0.00 0.00 0.00;
         0.00 0.00 0.00;
         0.00 0.00 0.00];

    %% Input signal: 3 channels
    u = zeros(N,3);
    u(:,1) = 0.8*sin(2*pi*0.15*t) + 0.4*sin(2*pi*0.60*t + 0.4) + 0.2*randn(N,1);
    u(:,2) = 0.7*sin(2*pi*0.20*t + 0.7) + 0.5*sin(2*pi*0.45*t) + 0.2*randn(N,1);
    u(:,3) = 0.9*sin(2*pi*0.10*t + 0.2) + 0.3*sin(2*pi*0.90*t + 0.8) + 0.2*randn(N,1);

    %% Noise settings
    sigmaW = 0.01;   % process noise std
    sigmaV = 0.02;   % measurement noise std

    %% Simulate
    nx = size(A,1);
    ny = size(C,1);

    x = zeros(N, nx);
    y = zeros(N, ny);

    for k = 1:N-1
        w = sigmaW * randn(nx,1);
        v = sigmaV * randn(ny,1);

        y(k,:)   = (C*x(k,:).' + D*u(k,:).' + v).';
        x(k+1,:) = (A*x(k,:).' + B*u(k,:).' + w).';
    end
    y(N,:) = (C*x(N,:).' + D*u(N,:).' + sigmaV*randn(ny,1)).';

    %% Suggested MIMO NARX orders
    na = [2 2 2;   % output 1 depends on past y1,y2,y3
          2 2 2;
          2 2 2];

    nb = [2 2 2;   % each output uses all 3 inputs
          2 2 2;
          2 2 2];

    nk = [1 1 1;
          1 1 1;
          1 1 1];

    %% Train/test split
    Ntr = round(0.7 * N);

    data = struct();
    data.modelName = '3x3 Linear MIMO Example';
    data.Ts = Ts;
    data.t = t;

    data.A = A;
    data.B = B;
    data.C = C;
    data.D = D;

    data.x = x;
    data.u = u;   % N x 3
    data.y = y;   % N x 3

    data.na = na;
    data.nb = nb;
    data.nk = nk;

    data.uEst = u(1:Ntr,:);
    data.yEst = y(1:Ntr,:);
    data.uVal = u(Ntr+1:end,:);
    data.yVal = y(Ntr+1:end,:);

    data.idxVal = (Ntr+1:N).';
end