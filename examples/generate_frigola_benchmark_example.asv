function data = generate_frigola_benchmark_example()
% Generate GP-NARX training/test data for the nonlinear benchmark system
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
%   (a,b,c,d,q,r) = (0.5, 25, 8, 0.05, 10, 1)
%   u(k) = cos(1.2*(k+1))

    rng(1);

    %% Parameters
    a = 0.5;
    b = 25;
    c = 8;
    d = 0.05;
    q = 0.1;
    r = 0.01;

    %% Simulation settings
    N  = 500;
    Ts = 0.1;
    t  = (0:N-1)' * Ts;

    %% Input
    u = cos(1.2 * (t + 1));

    %% Simulate latent state and measured output
    x = zeros(N+1, 1);
    y = zeros(N, 1);

    x(1) = 0;

    for k = 1:N
        v = sqrt(q) * randn;
        e = sqrt(r) * randn;

        x(k+1) = a*x(k) + b*x(k)/(1 + x(k)^2) + c*u(k) + v;
        y(k)   = d*x(k)^2 + e;
    end

    %% NARX settings
    % Since y_t depends nonlinearly on latent x_t and the system is dynamic,
    % a slightly larger model order is helpful.
    na = 15;
    nb = 15;
    nk = 1;

    [X, Y, idxMap] = build_narx_regressors(y, u, na, nb, nk);

    %% Train/test split
    Ndata = size(X,1);
    Ntr = round(0.7 * Ndata);

    %% Pack output
    data = struct();
    data.modelName = 'Frigola Nonlinear Benchmark';
    data.Ts = Ts;
    data.t = t;
    data.u = u;
    data.y = y;
    data.x = x(1:N);        % latent state aligned with y
    data.x_next = x(2:N+1); % optional next state

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

    %% Save true parameters for reference
    data.trueParams = struct('a',a,'b',b,'c',c,'d',d,'q',q,'r',r);
end