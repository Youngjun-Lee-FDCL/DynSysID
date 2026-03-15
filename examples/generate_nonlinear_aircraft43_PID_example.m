function data = generate_nonlinear_aircraft43_PID_example()
% Generate data for a nonlinear 4-input 3-output aircraft-like system
% using PID controllers for step-reference tracking.
%
% States:
%   x1 = q     : pitch rate
%   x2 = p     : roll rate
%   x3 = r     : yaw rate
%   x4 = theta : pitch angle
%   x5 = phi   : roll angle
%   x6 = V     : speed
%
% Inputs:
%   u1 = de : elevator
%   u2 = da : aileron
%   u3 = dr : rudder
%   u4 = dt : thrust command
%
% Outputs:
%   y1 = theta
%   y2 = phi
%   y3 = r
%
% Output:
%   data struct containing train/validation split and metadata

    rng(1);

    %% Simulation settings
    Ts = 0.05;
    N  = 1500;
    t  = (0:N-1)' * Ts;

    %% Noise settings
    sigmaW = 0.01*[0.003 0.003 0.003 0.002 0.002 0.010];
    sigmaV = 0.01*[0.005 0.005 0.005];

    %% PID gains
    % y1=theta -> elevator
    % y2=phi   -> aileron
    % y3=r     -> rudder
    %
    % thrust is coupled weakly to theta and phi loops
    Kp = [2.8; 2.5; 1.8];
    Ki = [0.25; 0.20; 0.12];
    Kd = [0.12; 0.10; 0.08];

    %% Input limits
    % [de, da, dr, dt]
    uMin = [-0.4, -0.4, -0.4, 0.1];
    uMax = [ 0.4,  0.4,  0.4, 1.0];

    %% Input bias
    uBias = [0.0, 0.0, 0.0, 0.5];

    %% References for outputs [theta, phi, r]
    r = zeros(N,3);
    changeTimes = [0 12 24 36 48 60];
    r(:,1) = make_step_profile(t, changeTimes, [ 0.00  0.08 -0.06  0.10 -0.08  0.05]);
    r(:,2) = make_step_profile(t, changeTimes, [ 0.00 -0.10  0.08 -0.12  0.10 -0.06]);
    r(:,3) = make_step_profile(t, changeTimes, [ 0.00  0.05 -0.04  0.06 -0.05  0.03]);

    %% Dimensions
    nx = 6;
    ny = 3;
    nu = 4;

    %% Preallocation
    x = zeros(N, nx);
    y = zeros(N, ny);
    u = zeros(N, nu);

    eInt  = zeros(1,ny);
    ePrev = zeros(1,ny);

    %% Initial state
    % [q p r theta phi V]
    x(1,:) = [0 0 0 0 0 100];
    u(1,:) = uBias;

    y(1,:) = [x(1,4), x(1,5), x(1,3)] + sigmaV .* randn(1,ny);

    %% Closed-loop simulation
    for k = 1:N-1
        % Outputs
        yk = y(k,:);

        % Tracking error
        e = r(k,:) - yk;

        % PID terms
        eInt = eInt + e * Ts;
        eDer = (e - ePrev) / Ts;

        pidOut = (Kp .* e.' + Ki .* eInt.' + Kd .* eDer.').';

        % Input mapping
        % de <- theta loop
        % da <- phi loop
        % dr <- r loop
        % dt <- weak coupling to theta/phi loop
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

        %% State extraction
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

        %% Nonlinear aircraft-like dynamics
        %
        % q_dot, p_dot, r_dot include damping, cross-coupling,
        % nonlinear aerodynamic terms, and control inputs.
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

        %% Euler update + process noise
        w = sigmaW .* randn(1,nx);

        x(k+1,1) = q     + Ts*qDot     + w(1);
        x(k+1,2) = p     + Ts*pDot     + w(2);
        x(k+1,3) = rYaw  + Ts*rDot     + w(3);
        x(k+1,4) = theta + Ts*thetaDot + w(4);
        x(k+1,5) = phi   + Ts*phiDot   + w(5);
        x(k+1,6) = V     + Ts*VDot     + w(6);

        % Optional clipping for realism
        x(k+1,4) = min(max(x(k+1,4), -0.7), 0.7);
        x(k+1,5) = min(max(x(k+1,5), -0.8), 0.8);
        x(k+1,6) = max(x(k+1,6), 40);

        %% Output equation
        v = sigmaV .* randn(1,ny);
        y(k+1,:) = [x(k+1,4), x(k+1,5), x(k+1,3)] + v;

        ePrev = e;
    end

    % Final input sample
    u(N,:) = u(N-1,:);

    %% Suggested model orders
    na = 3 * ones(3,3);
    nb = 3 * ones(3,4);
    nk = 1 * ones(3,4);

    %% Train/validation split
    Ntr = round(0.7 * N);

    data = struct();
    data.modelName = 'Nonlinear Aircraft-like 4x3 PID Tracking Example';
    data.Ts = Ts;
    data.t  = t;

    data.x = x;
    data.r = r;
    data.u = u;
    data.y = y;

    data.uBias = uBias;

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