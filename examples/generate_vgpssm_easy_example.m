function data = generate_vgpssm_easy_example()

    rng(1)

    T  = 500;
    Dx = 2;

    dt = 0.1;
    t  = (0:T-1)' * dt;
    
    sigmaU = 0.02;
    u = sin(0.4*t) + sigmaU*randn(T,1);

    x = zeros(T,Dx);

    for k = 1:T-1
        x1 = x(k,1);
        x2 = x(k,2);

        x(k+1,1) = x1 + dt*x2;
        x(k+1,2) = 0.95*x2 + 0.2*sin(x1) + 0.1*u(k);
    end
    
    sigmaY = 0.005;
    y = x + sigmaY*randn(T,Dx);

    data.t = t;
    data.u = u;
    data.x = x;
    data.y = y;

    data.na = 10;
    data.nb = 10;
    data.nk = 1;
    data.modelName = 'vgpssm_easy';

    data.Ts = dt;
end