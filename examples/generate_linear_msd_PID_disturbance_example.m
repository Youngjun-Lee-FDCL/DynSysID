function data = generate_linear_msd_PID_disturbance_example(numExp, seed)
% Generate multi-experiment data for a mass-spring-damper plant
% under PID control with constant disturbance.
%
% Plant:
%   m*xdd + c*xd + k*x = u + d
%
% Identification I/O:
%   input  = [u, r]
%   output = [y, I_rec]
%
% Returned fields are compatible with the existing exampleFcn(numExp) style.

    if nargin < 1 || isempty(numExp)
        numExp = 5;
    end
    if nargin >= 2 && ~isempty(seed)
        rng(seed);
    end

    %% =========================================================
    % 1) Global settings
    %% =========================================================
    Ts   = 0.02;
    Tend = 80;
    t    = (0:Ts:Tend)';
    N    = numel(t);

    modelName = 'Linear MSD with PID and constant disturbance';

    % Recommended NARX orders for this example
    na = 3;
    nb = 3;
    nk = 1;

    %% =========================================================
    % 2) True plant: mass-spring-damper with constant disturbance
    %% =========================================================
    m = 1.0;
    c = 1.2;
    k = 20.0;

    d_const = 0.0;

    Ac = [0 1;
         -k/m -c/m];
    Bc = [0;
          1/m];
    Ec = [0;
          1/m];
    Cc = [1 0];
    Dc = 0;

    sysc_u = ss(Ac, Bc, Cc, Dc);
    sysd_u = c2d(sysc_u, Ts);

    Ad = sysd_u.A;
    Bd = sysd_u.B;
    Cd = sysd_u.C;

    sysc_d = ss(Ac, Ec, Cc, 0);
    sysd_d = c2d(sysc_d, Ts);
    Ed = sysd_d.B;

    %% =========================================================
    % 3) Saturation and measurement noise
    %% =========================================================
    u_min   = -15;
    u_max   =  15;
    sigma_y = 0.000;

    %% =========================================================
    % 4) Nominal PID gains and variability
    %    Small randomization across experiments improves diversity
    %% =========================================================
    Kp_nom = 150;
    Ki_nom = 80;
    Kd_nom = 50;

    pidRandScale = [0.20, 0.20, 0.20];   % relative perturbation

    %% =========================================================
    % 5) Train/validation split
    %% =========================================================
    estRatio = 0.70;
    idxSplit = max(2, min(N-1, floor(estRatio * N)));

    %% =========================================================
    % 6) Preallocate cells
    %% =========================================================
    tCell    = cell(numExp,1);
    uCell    = cell(numExp,1);
    yCell    = cell(numExp,1);

    uEstCell = cell(numExp,1);
    yEstCell = cell(numExp,1);
    uValCell = cell(numExp,1);
    yValCell = cell(numExp,1);

    idxEstCell = cell(numExp,1);
    idxValCell = cell(numExp,1);

    refCell     = cell(numExp,1);
    pidGainCell = cell(numExp,1);
    truthCell   = cell(numExp,1);

    %% =========================================================
    % 7) Generate experiments
    %% =========================================================
    for e = 1:numExp
        % Random PID gains around nominal values
        Kp = Kp_nom * (1 + pidRandScale(1) * (2*rand - 1));
        Ki = Ki_nom * (1 + pidRandScale(2) * (2*rand - 1));
        Kd = Kd_nom * (1 + pidRandScale(3) * (2*rand - 1));

        % Random reference profile
        r = generate_piecewise_reference_msd(t, e);

        % Closed-loop simulation
        simData = simulate_closed_loop_pid_msd( ...
            Ad, Bd, Cd, Ed, d_const, ...
            Ts, N, r, ...
            Kp, Ki, Kd, ...
            u_min, u_max, sigma_y);

        % Identification input/output
        U = [simData.u, r];
        Y = [simData.y, simData.I_rec];

        idxEst = (1:idxSplit).';
        idxVal = (idxSplit+1:N).';

        % Store
        tCell{e} = t;
        uCell{e} = U;
        yCell{e} = Y;

        uEstCell{e} = U(idxEst,:);
        yEstCell{e} = Y(idxEst,:);
        uValCell{e} = U(idxVal,:);
        yValCell{e} = Y(idxVal,:);

        idxEstCell{e} = idxEst;
        idxValCell{e} = idxVal;

        refCell{e} = r;
        pidGainCell{e} = [Kp, Ki, Kd];

        truthCell{e} = struct( ...
            'x', simData.x, ...
            'y', simData.y, ...
            'u', simData.u, ...
            'e', simData.e, ...
            'I', simData.I, ...
            'I_rec', simData.I_rec, ...
            'e_dot', simData.e_dot);
    end

    %% =========================================================
    % 8) Output struct
    %% =========================================================
    data = struct();

    data.modelName = modelName;
    data.numExp    = numExp;
    data.Ts        = Ts;
    data.t         = t;

    data.na = na;
    data.nb = nb;
    data.nk = nk;

    data.nu = 2;   % [u, r]
    data.ny = 2;   % [y, I_rec]

    data.inputName  = {'u', 'r'};
    data.outputName = {'y', 'I_rec'};

    data.tCell    = tCell;
    data.uCell    = uCell;
    data.yCell    = yCell;

    data.uEstCell = uEstCell;
    data.yEstCell = yEstCell;
    data.uValCell = uValCell;
    data.yValCell = yValCell;

    data.idxEstCell = idxEstCell;
    data.idxValCell = idxValCell;

    data.refCell     = refCell;
    data.pidGainCell = pidGainCell;
    data.truthCell   = truthCell;

    data.description = [ ...
        "Mass-spring-damper plant with constant disturbance under PID control. " ...
        "Identification input is [u, r], and output is [y, I_rec]."];
end


function r = generate_piecewise_reference_msd(t, expIdx)
% Generate a piecewise-constant reference with mild sinusoidal perturbation.

    N = numel(t);
    Tfinal = t(end);

    % Segment boundaries
    tau1 = 0.05 * Tfinal;
    tau2 = 0.25 * Tfinal;
    tau3 = 0.45 * Tfinal;
    tau4 = 0.70 * Tfinal;
    tau5 = 0.85 * Tfinal;

    % Experiment-dependent base levels
    baseLevels = [ ...
         0.00,  0.50, -0.80,  0.50,  0.20; ...
         0.00,  0.35, -0.45,  0.65, -0.10; ...
         0.00,  0.60, -0.20,  0.70,  0.10; ...
         0.00,  0.45, -0.35,  0.55,  0.25; ...
         0.00,  0.30, -0.50,  0.75,  0.40];

    rowIdx = mod(expIdx-1, size(baseLevels,1)) + 1;
    lvl = baseLevels(rowIdx,:);

    r = zeros(N,1);
    r(t >= tau1 & t < tau2) = lvl(2);
    r(t >= tau2 & t < tau3) = lvl(3);
    r(t >= tau3 & t < tau4) = lvl(4);
    r(t >= tau4)            = lvl(5);   
end


function simData = simulate_closed_loop_pid_msd( ...
    Ad, Bd, Cd, Ed, d_const, ...
    Ts, N, r, ...
    Kp, Ki, Kd, ...
    u_min, u_max, sigma_y)
% Simulate closed-loop PID control with anti-windup clamping.

    nx = size(Ad,1);

    x     = zeros(nx, N);
    y     = zeros(N,1);
    u     = zeros(N,1);
    e     = zeros(N,1);
    I     = zeros(N,1);
    I(1) = 1;
    e_dot = zeros(N,1);

    for k_idx = 1:N-1
        % Current measured output
        y(k_idx) = Cd * x(:,k_idx) + sigma_y * randn;

        % Tracking error
        e(k_idx) = r(k_idx) - y(k_idx);

        % Error derivative
        if k_idx == 1
            e_dot(k_idx) = 0;
        else
            e_dot(k_idx) = (e(k_idx) - e(k_idx-1)) / Ts;
        end

        % Candidate integrator update
        I_candidate = I(k_idx) + Ts * e(k_idx);

        % Unsaturated PID command
        u_unsat = Kp * e(k_idx) + Ki * I_candidate + Kd * e_dot(k_idx);

        % Anti-windup clamping
        if (u_unsat > u_max && e(k_idx) > 0) || (u_unsat < u_min && e(k_idx) < 0)
            I(k_idx+1) = I(k_idx);
        else
            I(k_idx+1) = I_candidate;
        end
        
      
        
        % Final saturated control
        u_pid = Kp * e(k_idx) + Ki * I(k_idx+1) + Kd * e_dot(k_idx);
        u_pid = 5.8*u_pid;
        u(k_idx) = min(max(u_pid, u_min), u_max);
        
        % step-command disturbance
        if k_idx > floor(N/2)
            dist = d_const;
        else
            dist = 0;
        end

        % Plant update
        x(:,k_idx+1) = Ad * x(:,k_idx) + Bd * u(k_idx) + Ed * dist;
    end

    % Final sample
    y(N) = Cd * x(:,N) + sigma_y * randn;
    e(N) = r(N) - y(N);

    if N >= 2
        e_dot(N) = (e(N) - e(N-1)) / Ts;
        u(N)     = u(N-1);
    end

    % Reconstruct integral state from measured data only
    I_rec = zeros(N,1);
    for k_idx = 1:N-1
        err_k = r(k_idx) - y(k_idx);
        I_candidate = I_rec(k_idx) + Ts * err_k;

        if k_idx == 1
            e_dot_rec = 0;
        else
            err_prev  = r(k_idx-1) - y(k_idx-1);
            e_dot_rec = (err_k - err_prev) / Ts;
        end

        u_unsat_rec = Kp * err_k + Ki * I_candidate + Kd * e_dot_rec;

        if (u_unsat_rec > u_max && err_k > 0) || (u_unsat_rec < u_min && err_k < 0)
            I_rec(k_idx+1) = I_rec(k_idx);
        else
            I_rec(k_idx+1) = I_candidate;
        end
    end

    simData = struct();
    simData.x     = x;
    simData.y     = y;
    simData.u     = u;
    simData.e     = e;
    simData.I     = I;
    simData.I_rec = I_rec;
    simData.e_dot = e_dot;
end