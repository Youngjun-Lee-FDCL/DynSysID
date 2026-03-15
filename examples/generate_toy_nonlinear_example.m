function data = generate_toy_nonlinear_example()
% Generate GP-NARX training/test data for the original toy nonlinear example
%
% System:
% y(k) = 0.7*y(k-1)/(1+y(k-1)^2) + 0.4*u(k-1) + 0.2*u(k-2)^2 + noise

    rng(1);

    N = 400;
    Ts = 1.0;
    t = (0:N-1)' * Ts;

    u = randn(N,1);
    y = zeros(N,1);

    sigmaNoise = 0.05;

    for k = 3:N
        y(k) = 0.7 * y(k-1) / (1 + y(k-1)^2) ...
             + 0.4 * u(k-1) ...
             + 0.2 * u(k-2)^2 ...
             + sigmaNoise * randn;
    end

    % NARX settings
    na = 2;
    nb = 2;
    nk = 1;

    [X, Y, idxMap] = build_narx_regressors(y, u, na, nb, nk);

    % split
    Ndata = size(X,1);
    Ntr = round(0.7 * Ndata);

    data = struct();
    data.modelName = 'Toy Nonlinear Example';
    data.Ts = Ts;
    data.t = t;
    data.u = u;
    data.y = y;
    data.x = [];   % no physical state here

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
end