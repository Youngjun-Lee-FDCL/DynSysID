function data = generate_nonlinear_msd_example()
% Generate GP-NARX training/test data for nonlinear mass-spring-damper system

    rng(1);

    Ts = 0.02;
    N  = 800;
    t  = (0:N-1)' * Ts;

    % input
    sigmaU = 0.5;
    u = 1.0*sin(2*pi*0.4*t) ...
      + 0.7*sin(2*pi*1.1*t + 0.3) ...
      + sigmaU*randn(N,1);

    % parameters
    m  = 1.0;
    c  = 0.8;
    k  = 20.0;
    c3 = 0.4;
    k3 = 15.0;

    sigmaY = 0.003;

    % simulate
    x = zeros(2,N);
    y = zeros(N,1);

    for kk = 1:N-1
        x(:,kk+1) = rk4_step_nonlinear(x(:,kk), u(kk), Ts, m, c, k, c3, k3);
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
    data.modelName = 'Nonlinear MSD';
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

function xNext = rk4_step_nonlinear(x, u, Ts, m, c, k, c3, k3)
    k_1 = f_nonlinear(x, u, m, c, k, c3, k3);
    k_2 = f_nonlinear(x + 0.5*Ts*k_1, u, m, c, k, c3, k3);
    k_3 = f_nonlinear(x + 0.5*Ts*k_2, u, m, c, k, c3, k3);
    k_4 = f_nonlinear(x + Ts*k3, u, m, c, k, c3, k3);

    xNext = x + (Ts/6)*(k_1 + 2*k_2 + 2*k_3 + k_4);
end

function dx = f_nonlinear(x, u, m, c, k, c3, k3)
    pos = x(1);
    vel = x(2);
    acc = (u - c*vel - c3*vel^3 - k*pos - k3*pos^3)/m;
    dx = [vel; acc];
end