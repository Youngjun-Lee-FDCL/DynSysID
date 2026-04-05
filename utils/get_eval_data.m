function [uEval, yEval, tEval] = get_eval_data(data, Ts, expIdx, evalMode)

    switch lower(evalMode)
        case 'val'
            [uEval, yEval] = force_2d_io(data.uValCell{expIdx}, data.yValCell{expIdx});

            if isfield(data, 'tCell') && isfield(data, 'idxValCell')
                tEval = data.tCell{expIdx}(data.idxValCell{expIdx});
            else
                tEval = (0:size(yEval,1)-1)' * Ts;
            end

        case 'full'
            [uEval, yEval] = force_2d_io(data.uCell{expIdx}, data.yCell{expIdx});

            if isfield(data, 'tCell')
                tEval = data.tCell{expIdx};
            else
                tEval = (0:size(yEval,1)-1)' * Ts;
            end

        otherwise
            error('evalMode must be ''val'' or ''full''.');
    end
end

function [model, info] = build_smc_model_from_n4sid(sysN4, uEval, yEval, Ts, useFindstatesForX0, P0Scale)
% Build model struct compatible with SMCSmoother
%
% State model:
%   x_{t+1} = A x_t + B u_t + w_t
%   y_t     = C x_t + D u_t + v_t
%
% where
%   w_t ~ N(0,Q), v_t ~ N(0,R)

    A = sysN4.A;
    B = sysN4.B;
    C = sysN4.C;
    D = sysN4.D;
    nx = size(A,1);

    % Observation noise covariance
    R = sysN4.NoiseVariance;
    if isempty(R)
        error('sysN4.NoiseVariance is empty.');
    end
    R = 0.5 * (R + R.');
    R = regularize_cov(R, 1e-8);

    % Innovation gain
    try
        K = sysN4.K;
    catch
        error('sysN4.K could not be accessed.');
    end

    % Practical process covariance
    Q = K * R * K.';
    Q = 0.5 * (Q + Q.');
    Q = regularize_cov(Q, 1e-10);

    % Initial mean
    if useFindstatesForX0
        zEval = iddata(yEval, uEval, Ts);
        x0Mean = findstates(sysN4, zEval, Inf);
        x0Mean = x0Mean(:);
    else
        x0Mean = zeros(nx,1);
    end

    % Initial covariance
    P0 = P0Scale * eye(nx);
    P0 = regularize_cov(P0, 1e-12);

    cholP0 = chol(P0, 'lower');
    cholQ  = chol(Q, 'lower');
    invR   = inv(R);
    logDetR = log(det(R));
    logDetQ = log(det(Q));
    ny = size(C,1);

    T = size(yEval,1); %#ok<NASGU>

    % Internal time index for transition sampling
    tCounter = 0;

    model = struct();

    model.sampleX0 = @sampleX0;
    model.sampleTransition = @sampleTransition;
    model.logLikelihood = @logLikelihood;
    model.logTransitionPdf = @logTransitionPdf;

    info = struct();
    info.A = A;
    info.B = B;
    info.C = C;
    info.D = D;
    info.Q = Q;
    info.R = R;
    info.P0 = P0;
    info.x0Mean = x0Mean;
    info.nx = nx;

    function x = sampleX0(N)
        tCounter = 1;
        x = x0Mean + cholP0 * randn(nx, N);
    end

    function x = sampleTransition(xPrev)
        % xPrev : [nx x N]
        % transition from time tCounter to tCounter+1
        % uses u(tCounter,:)

        [nx_, N] = size(xPrev); %#ok<ASGLU>
        if tCounter < 1
            error('sampleX0 must be called before sampleTransition.');
        end

        if tCounter > size(uEval,1)
            error('Transition time index exceeded data length.');
        end

        uNow = uEval(tCounter,:).';   % input at current time
        x = A * xPrev + B * repmat(uNow, 1, N) + cholQ * randn(nx, N);

        tCounter = tCounter + 1;
    end

    function logw = logLikelihood(y_t, x_t)
        % y_t : [1 x ny]
        % x_t : [nx x N]
        % log p(y_t | x_t)

        N = size(x_t, 2);
        yVec = y_t(:);
        yMean = C * x_t + D * repmat(y_input_for_ll(), 1, N);

        diff = yVec - yMean;
        quad = sum(diff .* (invR * diff), 1);

        logw = -0.5 * (ny * log(2*pi) + logDetR + quad);
        logw = logw(:);
    end

    function logp = logTransitionPdf(x_t, xPrev)
        % x_t   : [nx x 1]
        % xPrev : [nx x N]
        % log p(x_t | xPrev)

        % backwardFFBSi is called after forward ended.
        % At that point, exact time index is not passed in.
        % For LTI system with known input sequence, we need transition time.
        % Since standard FFBSi in this class does not pass t explicitly,
        % we approximate using autonomous part only if input influence is small,
        % OR assume input enters deterministically through stored current time.
        %
        % To make FFBSi exact for input-driven systems, we use an augmented trick:
        % x_{t+1} - B u_t ~ N(A x_t, Q)
        %
        % Here, backwardFFBSi calls this function at each t internally,
        % but does not provide t. Therefore, we use persistent counter reset
        % through a wrapper below instead of direct indexing.
        %
        % Since the class interface lacks t, we define this function only for
        % use with a wrapper function outside if needed. For now, we use the
        % autonomous approximation:
        %
        %   p(x_{t+1} | x_t) ≈ N(A x_t, Q)
        %
        % If you want exact input-aware FFBSi, the class interface should be
        % modified to pass time index t.

        N = size(xPrev, 2);
        diff = x_t - A * xPrev;
        quad = sum(diff .* (Q \ diff), 1);

        logp = -0.5 * (nx * log(2*pi) + logDetQ + quad);
        logp = logp(:);
    end

    function uNow = y_input_for_ll()
        % logLikelihood is called sequentially in forward(), so we can infer time:
        % after sampleX0 -> y(1)
        % after sampleTransition -> y(t)
        %
        % Since sampleTransition increments tCounter after propagation,
        % at likelihood evaluation current observation time is tCounter.
        if tCounter < 1
            idx = 1;
        else
            idx = min(tCounter, size(uEval,1));
        end
        uNow = uEval(idx,:).';
    end
end

function S = regularize_cov(S, jitter)
    S = 0.5 * (S + S.');
    [~, p] = chol(S);
    if p ~= 0
        S = S + jitter * eye(size(S));
    end
end

function [u, y] = force_2d_io(u, y)
    if isvector(u), u = u(:); end
    if isvector(y), y = y(:); end
end

function x = force_2d_state(x)
    if isvector(x), x = x(:); end
end