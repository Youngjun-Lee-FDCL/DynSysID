function data = generate_linear_msd_example()
% Generate GP-NARX training/test data for linear mass-spring-damper system

    rng(1);

    Ts = 0.02;
    N  = 800;
    t  = (0:N-1)' * Ts;

    % input
    u = 1.0*sin(2*pi*0.4*t) ...
      + 0.7*sin(2*pi*1.1*t + 0.3) ...
      + 0.3*randn(N,1);

    % parameters
    m = 1.0;
    c = 0.8;
    k = 20.0;

    sigmaY = 0.002;

    % simulate
    x = zeros(2,N);
    y = zeros(N,1);

    for kk = 1:N-1
        x(:,kk+1) = rk4_step_linear(x(:,kk), u(kk), Ts, m, c, k);
        y(kk) = x(1,kk) + sigmaY*randn;
    end
    y(N) = x(1,N) + sigmaY*randn;

    % NARX settings
    na = 10;
    nb = 10;
    nk = 1;

    [X, Y, idxMap] = build_narx_regressors(y, u, na, nb, nk);

    % split
    Ndata = size(X,1);
    Ntr = round(0.7*Ndata);

    data = struct();
    data.modelName = 'Linear MSD';
    data.Ts = Ts;
    data.t = t;
    data.u = u;
    data.y = y;
    data.x = x;

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

function xNext = rk4_step_linear(x, u, Ts, m, c, k)
    k1 = f_linear(x, u, m, c, k);
    k2 = f_linear(x + 0.5*Ts*k1, u, m, c, k);
    k3 = f_linear(x + 0.5*Ts*k2, u, m, c, k);
    k4 = f_linear(x + Ts*k3, u, m, c, k);

    xNext = x + (Ts/6)*(k1 + 2*k2 + 2*k3 + k4);
end

function dx = f_linear(x, u, m, c, k)
    pos = x(1);
    vel = x(2);
    acc = (u - c*vel - k*pos)/m;
    dx = [vel; acc];
end