classdef SMCSmoother
    properties
        NumParticles   (1,1) double {mustBePositive, mustBeInteger} = 100
        StateDim       (1,1) double {mustBePositive, mustBeInteger} = 1
        ResampleThreshold (1,1) double = 0.5   % ESS/N threshold
    end

    methods
        function obj = SMCSmoother(numParticles, stateDim)
            if nargin >= 1
                obj.NumParticles = numParticles;
            end
            if nargin >= 2
                obj.StateDim = stateDim;
            end
        end

        function out = forward(obj, y, model)
            % FORWARD  Bootstrap particle filter
            %
            % Inputs
            %   y     : [T x ny] observation sequence
            %   model : struct with fields
            %       .sampleX0(N)                -> [nx x N]
            %       .sampleTransition(xPrev)    -> [nx x N]
            %       .logLikelihood(y_t, x_t)    -> [1 x N] or [N x 1]
            %
            % Output
            %   out.particles  : [nx x N x T]
            %   out.weights    : [N x T] normalized filtering weights
            %   out.ancestors  : [N x T] ancestor indices after resampling
            %   out.ess        : [T x 1] effective sample size
            %   out.xFiltMean  : [nx x T] filtering mean estimate

            arguments
                obj
                y double
                model struct
            end

            % Basic checks
            requiredFields = {'sampleX0', 'sampleTransition', 'logLikelihood'};
            for k = 1:numel(requiredFields)
                if ~isfield(model, requiredFields{k})
                    error('model.%s is required.', requiredFields{k});
                end
            end

            T = size(y, 1);
            N = obj.NumParticles;
            nx = obj.StateDim;

            particles = zeros(nx, N, T);
            weights   = zeros(N, T);
            ancestors = zeros(N, T);
            essHist   = zeros(T, 1);
            xFiltMean = zeros(nx, T);

            % -----------------------------------------------------------------
            % t = 1 : initialize from p(x0), then weight with y(1)
            % -----------------------------------------------------------------
            x = model.sampleX0(N);   % expected size [nx x N]
            if ~isequal(size(x), [nx, N])
                error('model.sampleX0(N) must return [%d x %d].', nx, N);
            end

            logw = model.logLikelihood(y(1, :), x);
            logw = reshape(logw, [], 1);
            if numel(logw) ~= N
                error('model.logLikelihood(y_t, x_t) must return N log-weights.');
            end

            w = obj.normalizeLogWeights(logw);
            ess = 1 / sum(w.^2);

            % optional resampling at t=1
            if ess < obj.ResampleThreshold * N
                idx = obj.systematicResample(w);
                x = x(:, idx);
                w = ones(N, 1) / N;
                ancestors(:, 1) = idx(:);
            else
                ancestors(:, 1) = (1:N)';
            end

            particles(:, :, 1) = x;
            weights(:, 1) = w;
            essHist(1) = ess;
            xFiltMean(:, 1) = x * w;

            % -----------------------------------------------------------------
            % t = 2:T
            % -----------------------------------------------------------------
            for t = 2:T
                xPrev = particles(:, :, t-1);
                wPrev = weights(:, t-1);

                % Bootstrap PF:
                % resample first if needed, then propagate with transition
                essPrev = 1 / sum(wPrev.^2);

                if essPrev < obj.ResampleThreshold * N
                    a = obj.systematicResample(wPrev);
                    xParent = xPrev(:, a);
                else
                    a = 1:N;
                    xParent = xPrev;
                end
                ancestors(:, t) = a(:);

                % Propagate
                x = model.sampleTransition(xParent);
                if ~isequal(size(x), [nx, N])
                    error('model.sampleTransition(xPrev) must return [%d x %d].', nx, N);
                end

                % Weight with current observation
                logw = model.logLikelihood(y(t, :), x);
                logw = reshape(logw, [], 1);
                if numel(logw) ~= N
                    error('model.logLikelihood(y_t, x_t) must return N log-weights.');
                end

                w = obj.normalizeLogWeights(logw);
                ess = 1 / sum(w.^2);

                % Store current filtering approximation
                particles(:, :, t) = x;
                weights(:, t) = w;
                essHist(t) = ess;
                xFiltMean(:, t) = x * w;
            end

            out = struct();
            out.particles = particles;
            out.weights = weights;
            out.ancestors = ancestors;
            out.ess = essHist;
            out.xFiltMean = xFiltMean;
        end

        function traj = backward(obj, forwardOut)
            % BACKWARD
            % Sample one trajectory by tracing ancestors backward
            % from the final particle index drawn according to w_T.
            %
            % Input
            %   forwardOut : struct returned by forward()
            %       .particles  [nx x N x T]
            %       .weights    [N x T]
            %       .ancestors  [N x T]
            %
            % Output
            %   traj : [nx x T] sampled trajectory

            arguments
                obj
                forwardOut struct
            end

            requiredFields = {'particles', 'weights', 'ancestors'};
            for k = 1:numel(requiredFields)
                if ~isfield(forwardOut, requiredFields{k})
                    error('forwardOut.%s is required.', requiredFields{k});
                end
            end

            particles = forwardOut.particles;
            weights   = forwardOut.weights;
            ancestors = forwardOut.ancestors;

            [nx, N, T] = size(particles);

            if size(weights,1) ~= N || size(weights,2) ~= T
                error('weights must have size [N x T].');
            end

            if size(ancestors,1) ~= N || size(ancestors,2) ~= T
                error('ancestors must have size [N x T].');
            end

            traj = zeros(nx, T);

            % ------------------------------------------------------------
            % 1) Sample final particle index from filtering weights at time T
            % ------------------------------------------------------------
            idx = obj.sampleCategorical(weights(:, T));

            % ------------------------------------------------------------
            % 2) Set final state
            % ------------------------------------------------------------
            traj(:, T) = particles(:, idx, T);

            % ------------------------------------------------------------
            % 3) Trace ancestors backward
            % ------------------------------------------------------------
            for t = T:-1:2
                idx = ancestors(idx, t);
                traj(:, t-1) = particles(:, idx, t-1);
            end
        end

        function [traj, out] = pgasStep(obj, y, model, xRef)
            % PGASSTEP
            % One conditional SMC sweep with ancestor sampling.
            %
            % Inputs
            %   y     : [T x ny]
            %   model : struct with fields
            %       .sampleX0(N)                    -> [nx x N]
            %       .sampleTransition(xPrev)        -> [nx x N]
            %       .logLikelihood(y_t, x_t)        -> [N x 1] or [1 x N]
            %       .logTransitionPdf(x_t, xPrev)   -> [N x 1] or [1 x N]
            %                                         where x_t is [nx x 1],
            %                                         xPrev is [nx x N]
            %   xRef  : [nx x T] reference trajectory
            %
            % Outputs
            %   traj  : [nx x T] newly sampled trajectory
            %   out   : conditional particle system

            arguments
                obj
                y double
                model struct
                xRef double
            end

            requiredFields = {'sampleX0', 'sampleTransition', ...
                'logLikelihood', 'logTransitionPdf'};
            for k = 1:numel(requiredFields)
                if ~isfield(model, requiredFields{k})
                    error('model.%s is required for PGAS.', requiredFields{k});
                end
            end

            T  = size(y, 1);
            N  = obj.NumParticles;
            nx = obj.StateDim;

            if ~isequal(size(xRef), [nx, T])
                error('xRef must have size [%d x %d].', nx, T);
            end

            particles = zeros(nx, N, T);
            weights   = zeros(N, T);
            ancestors = zeros(N, T);
            essHist   = zeros(T, 1);
            xFiltMean = zeros(nx, T);

            % ------------------------------------------------------------
            % t = 1
            % First N-1 particles sampled freely, last particle fixed to ref
            % ------------------------------------------------------------
            if N < 2
                error('PGAS requires NumParticles >= 2.');
            end

            xFree = model.sampleX0(N-1);
            if ~isequal(size(xFree), [nx, N-1])
                error('model.sampleX0(N-1) must return [%d x %d].', nx, N-1);
            end

            x = [xFree, xRef(:,1)];

            logw = model.logLikelihood(y(1,:), x);
            logw = reshape(logw, [], 1);
            if numel(logw) ~= N
                error('model.logLikelihood(y_t, x_t) must return N log-weights.');
            end

            w = obj.normalizeLogWeights(logw);

            particles(:,:,1) = x;
            weights(:,1) = w;
            ancestors(:,1) = (1:N)';
            essHist(1) = 1 / sum(w.^2);
            xFiltMean(:,1) = x * w;

            % ------------------------------------------------------------
            % t = 2:T
            % Conditional SMC + ancestor sampling for the reference particle
            % ------------------------------------------------------------
            for t = 2:T
                xPrev = particles(:,:,t-1);
                wPrev = weights(:,t-1);

                % 1) Resample ancestors for free particles
                aFree = obj.sampleCategoricalMany(wPrev, N-1);

                % 2) Ancestor sampling for the reference particle
                %    Prob(a_t^N = i) ∝ w_{t-1}^i p(xRef_t | x_{t-1}^i)
                logAs = log(wPrev + realmin) + ...
                    reshape(model.logTransitionPdf(xRef(:,t), xPrev), [], 1);

                aRef = obj.sampleCategorical(obj.normalizeLogWeights(logAs));

                % 3) Propagate free particles
                xParentFree = xPrev(:, aFree);
                xFree = model.sampleTransition(xParentFree);
                if ~isequal(size(xFree), [nx, N-1])
                    error('model.sampleTransition(xPrev) must return [%d x %d].', nx, N-1);
                end

                % 4) Build particle set: free particles + conditioned reference particle
                x = [xFree, xRef(:,t)];

                % 5) Store ancestor indices
                ancestors(1:N-1, t) = aFree(:);
                ancestors(N, t)     = aRef;

                % 6) Weight with current observation
                logw = model.logLikelihood(y(t,:), x);
                logw = reshape(logw, [], 1);
                if numel(logw) ~= N
                    error('model.logLikelihood(y_t, x_t) must return N log-weights.');
                end

                w = obj.normalizeLogWeights(logw);

                particles(:,:,t) = x;
                weights(:,t) = w;
                essHist(t) = 1 / sum(w.^2);
                xFiltMean(:,t) = x * w;
            end

            % ------------------------------------------------------------
            % Sample final particle index and trace backward
            % ------------------------------------------------------------
            traj = zeros(nx, T);

            idx = obj.sampleCategorical(weights(:,T));
            traj(:,T) = particles(:,idx,T);

            for t = T:-1:2
                idx = ancestors(idx,t);
                traj(:,t-1) = particles(:,idx,t-1);
            end

            out = struct();
            out.particles = particles;
            out.weights = weights;
            out.ancestors = ancestors;
            out.ess = essHist;
            out.xFiltMean = xFiltMean;
        end

        function [traj, hist] = pgas(obj, y, model, xInit, numIter)
            % PGAS
            % Run multiple PGAS iterations.
            %
            % Inputs
            %   y       : [T x ny]
            %   model   : model struct (must include logTransitionPdf)
            %   xInit   : [nx x T] initial reference trajectory
            %             if empty, initialize by forward+backward
            %   numIter : number of PGAS iterations
            %
            % Outputs
            %   traj    : final sampled trajectory
            %   hist    : cell array of trajectories

            arguments
                obj
                y double
                model struct
                xInit double = []
                numIter (1,1) double {mustBePositive, mustBeInteger} = 10
            end

            T  = size(y,1);
            nx = obj.StateDim;

            if isempty(xInit)
                fwd = obj.forward(y, model);
                traj = obj.backward(fwd);
            else
                if ~isequal(size(xInit), [nx, T])
                    error('xInit must have size [%d x %d].', nx, T);
                end
                traj = xInit;
            end

            hist = cell(numIter,1);
            for k = 1:numIter
                [traj, out] = obj.pgasStep(y, model, traj);
                hist{k} = struct('traj', traj, 'ess', out.ess, 'xFiltMean', out.xFiltMean);
            end
        end

        function traj = backwardFFBSi(obj, forwardOut, model)
            % BACKWARDFFBSI
            % FFBSi: sample one trajectory using backward simulation
            %
            % Inputs
            %   forwardOut : struct returned by forward()
            %       .particles  [nx x N x T]
            %       .weights    [N x T]
            %   model : struct with field
            %       .logTransitionPdf(xNext, xCurr) -> [N x 1] or [1 x N]
            %           xNext : [nx x 1]
            %           xCurr : [nx x N]
            %
            % Output
            %   traj : [nx x T] sampled trajectory

            arguments
                obj
                forwardOut struct
                model struct
            end

            requiredFields = {'particles', 'weights'};
            for k = 1:numel(requiredFields)
                if ~isfield(forwardOut, requiredFields{k})
                    error('forwardOut.%s is required.', requiredFields{k});
                end
            end
            if ~isfield(model, 'logTransitionPdf')
                error('model.logTransitionPdf is required for FFBSi.');
            end

            particles = forwardOut.particles;
            weights   = forwardOut.weights;

            [nx, N, T] = size(particles);

            if size(weights,1) ~= N || size(weights,2) ~= T
                error('weights must have size [N x T].');
            end

            traj = zeros(nx, T);

            % ------------------------------------------------------------
            % 1) sample x_T ~ p(x_T | y_1:T) approximated by filtering particles
            % ------------------------------------------------------------
            idx = obj.sampleCategorical(weights(:,T));
            traj(:,T) = particles(:,idx,T);

            % ------------------------------------------------------------
            % 2) backward simulation
            %    sample x_t from weights proportional to
            %    w_t^i * p(x_{t+1} | x_t^i)
            % ------------------------------------------------------------
            for t = T-1:-1:1
                xCurr = particles(:,:,t);      % [nx x N]
                xNext = traj(:,t+1);           % [nx x 1]

                logBw = log(weights(:,t) + realmin) + ...
                    reshape(model.logTransitionPdf(xNext, xCurr), [], 1);

                bw = obj.normalizeLogWeights(logBw);
                idx = obj.sampleCategorical(bw);
                traj(:,t) = xCurr(:,idx);
            end
        end

        function [trajSamples, xSmoothMean] = backwardFFBSiMany(obj, forwardOut, model, numSamples)
            % BACKWARDFFBSIMANY
            % Draw multiple FFBSi trajectories and compute approximate smoothing mean
            %
            % Outputs
            %   trajSamples  : [nx x T x L]
            %   xSmoothMean  : [nx x T]

            arguments
                obj
                forwardOut struct
                model struct
                numSamples (1,1) double {mustBePositive, mustBeInteger} = 100
            end

            particles = forwardOut.particles;
            [nx, ~, T] = size(particles);

            trajSamples = zeros(nx, T, numSamples);
            for l = 1:numSamples
                trajSamples(:,:,l) = obj.backwardFFBSi(forwardOut, model);
            end

            xSmoothMean = mean(trajSamples, 3);
        end
    end

    methods (Static, Access = private)
        function idx = sampleCategorical(w)
            % SAMPLECATEGORICAL
            % Draw one sample from categorical distribution w

            w = w(:);
            c = cumsum(w);
            r = rand();
            idx = find(r <= c, 1, 'first');

            if isempty(idx)
                idx = numel(w);
            end
        end

        function idx = sampleCategoricalMany(w, M)
            % SAMPLECATEGORICALMANY
            % Draw M iid categorical samples from w

            w = w(:);
            c = cumsum(w);
            r = rand(M,1);
            idx = zeros(M,1);

            for m = 1:M
                j = find(r(m) <= c, 1, 'first');
                if isempty(j)
                    j = numel(w);
                end
                idx(m) = j;
            end
        end

        function w = normalizeLogWeights(logw)
            c = max(logw);
            w = exp(logw - c);
            s = sum(w);
            if ~(isfinite(s) && s > 0)
                w = ones(size(logw)) / numel(logw);
            else
                w = w / s;
            end
        end

        function idx = systematicResample(w)
            N = numel(w);
            edges = cumsum(w(:));
            edges(end) = 1.0;  % numerical safety

            u1 = rand() / N;
            u = u1 + (0:N-1)' / N;

            idx = zeros(N, 1);
            j = 1;
            for i = 1:N
                while u(i) > edges(j)
                    j = j + 1;
                end
                idx(i) = j;
            end
        end
    end
    methods (Static)
        function test(Q, R, T, N, seed)
            % TEST
            % Simple test for 1D random-walk state-space model:
            %   x_t = x_{t-1} + w_t,   w_t ~ N(0, Q)
            %   y_t = x_t + v_t,       v_t ~ N(0, R)
            %
            % Inputs
            %   Q    : state transition noise variance
            %   R    : observation noise variance
            %   T    : sequence length
            %   N    : number of particles
            %   seed : random seed
            %
            % Example
            %   SMCSmoother.test(1.0, 0.25, 50, 200, 1)

            % ------------------------------------------------------------
            % Defaults
            % ------------------------------------------------------------
            if nargin < 1 || isempty(Q),    Q = 1.0;   end
            if nargin < 2 || isempty(R),    R = 1.0;   end
            if nargin < 3 || isempty(T),    T = 50;    end
            if nargin < 4 || isempty(N),    N = 200;   end
            if nargin < 5 || isempty(seed), seed = 1;  end

            rng(seed);

            nx = 1;
            numFFBSiSamples = 200;
            numPGASIter = 20;

            if Q < 0 || R <= 0
                error('Q must satisfy Q >= 0 and R must satisfy R > 0.');
            end

            useTransitionPdf = (Q > 0);

            % ------------------------------------------------------------
            % Generate synthetic data
            % ------------------------------------------------------------
            xTrue = zeros(1, T);
            y = zeros(T, 1);

            xTrue(1) = randn();
            y(1) = xTrue(1) + sqrt(R) * randn();

            for t = 2:T
                xTrue(t) = xTrue(t-1) + sqrt(Q) * randn();
                y(t) = xTrue(t) + sqrt(R) * randn();
            end

            % ------------------------------------------------------------
            % Build model
            % ------------------------------------------------------------
            model = struct();
            model.sampleX0 = @(Np) randn(nx, Np);
            model.sampleTransition = @(xPrev) xPrev + sqrt(Q) * randn(size(xPrev));
            model.logLikelihood = @(y_t, x_t) ...
                -0.5 * log(2*pi*R) - 0.5 * ((y_t - x_t).^2) / R;

            if useTransitionPdf
                model.logTransitionPdf = @(x_t, xPrev) ...
                    -0.5 * log(2*pi*Q) - 0.5 * ((x_t - xPrev).^2) / Q;
            end

            % ------------------------------------------------------------
            % Forward filtering
            % ------------------------------------------------------------
            pf = SMCSmoother(N, nx);
            fwd = pf.forward(y, model);

            % Simple ancestor-tracing backward sample
            xBack = pf.backward(fwd);

            % ------------------------------------------------------------
            % FFBSi / PGAS (only when Q > 0)
            % ------------------------------------------------------------
            trajFFBSi   = [];
            xSmoothMean = [];
            xPGAS       = [];

            if useTransitionPdf
                trajFFBSi = pf.backwardFFBSi(fwd, model);
                [~, xSmoothMean] = pf.backwardFFBSiMany(fwd, model, numFFBSiSamples);
                [xPGAS, ~] = pf.pgas(y, model, xBack, numPGASIter);
            end

            % ------------------------------------------------------------
            % Kalman filter baseline
            % ------------------------------------------------------------
            [xKF, ~] = localKf1d(y, Q, R, 0, 1);

            % ------------------------------------------------------------
            % Diagnostics
            % ------------------------------------------------------------
            particleVar = squeeze(var(fwd.particles, 0, 2));   % [nx x T] -> here [1 x T]
            particleVar = reshape(particleVar, 1, []);

            rmseObs   = localRmse(y(:).', xTrue);
            rmseFilt  = localRmse(fwd.xFiltMean, xTrue);
            rmseKF    = localRmse(xKF, xTrue);
            rmseBack  = localRmse(xBack, xTrue);

            fprintf('====================================================\n');
            fprintf('SMCSmoother test\n');
            fprintf('Q    = %.6f\n', Q);
            fprintf('R    = %.6f\n', R);
            fprintf('T    = %d\n', T);
            fprintf('N    = %d\n', N);
            fprintf('seed = %d\n', seed);
            fprintf('----------------------------------------------------\n');
            fprintf('Final ESS             = %.2f\n', fwd.ess(end));
            fprintf('Final filtering mean  = %.6f\n', fwd.xFiltMean(end));
            fprintf('RMSE observation      = %.6f\n', rmseObs);
            fprintf('RMSE filtering mean   = %.6f\n', rmseFilt);
            fprintf('RMSE backward sample  = %.6f\n', rmseBack);
            fprintf('RMSE Kalman filter    = %.6f\n', rmseKF);

            if useTransitionPdf
                rmseFFBSi = localRmse(trajFFBSi, xTrue);
                rmseSmooth = localRmse(xSmoothMean, xTrue);
                rmsePGAS = localRmse(xPGAS, xTrue);

                fprintf('RMSE FFBSi sample     = %.6f\n', rmseFFBSi);
                fprintf('RMSE FFBSi mean       = %.6f\n', rmseSmooth);
                fprintf('RMSE PGAS trajectory  = %.6f\n', rmsePGAS);
            else
                fprintf('(FFBSi/PGAS skipped because Q = 0)\n');
            end
            fprintf('====================================================\n');

            % ------------------------------------------------------------
            % Plot: state trajectories
            % ------------------------------------------------------------
            figure('Name', 'State Estimation');
            hold on;
            plot(1:T, xTrue, 'k-',  'LineWidth', 1.5);
            plot(1:T, y,     'b:',  'LineWidth', 1.0);
            plot(1:T, fwd.xFiltMean, 'r--', 'LineWidth', 1.5);
            plot(1:T, xBack, 'g-.', 'LineWidth', 1.2);
            plot(1:T, xKF,   'm--', 'LineWidth', 1.5);

            legendEntries = {'True state', 'Observation', ...
                'PF filtering mean', 'Backward sample', 'Kalman filter'};

            if useTransitionPdf
                plot(1:T, trajFFBSi,   'y-.', 'LineWidth', 1.0);
                plot(1:T, xSmoothMean, 'c-',  'LineWidth', 1.5);
                plot(1:T, xPGAS,       'Color', [0.5 0 0.8], 'LineWidth', 1.3);

                legendEntries = [legendEntries, {'FFBSi sample', 'FFBSi mean', 'PGAS trajectory'}];
            end

            xlabel('t');
            ylabel('state');
            title(sprintf('SMCSmoother Test (Q = %.3g, R = %.3g, N = %d)', Q, R, N));
            legend(legendEntries, 'Location', 'best');
            grid on;
            hold off;

            % ------------------------------------------------------------
            % Plot: diversity diagnostics
            % ------------------------------------------------------------
            figure('Name', 'Particle Diversity');
            tiledlayout(3,1);

            nexttile;
            plot(1:T, fwd.ess, 'LineWidth', 1.3);
            ylabel('ESS');
            title('Particle Diversity Diagnostics');
            grid on;

            nexttile;
            plot(1:T, particleVar, 'LineWidth', 1.3);
            ylabel('Var(particles)');
            grid on;

            nexttile;
            plot(1:T, max(fwd.weights, [], 1), 'LineWidth', 1.3);
            xlabel('t');
            ylabel('max weight');
            grid on;

            % ============================================================
            % Local helper functions
            % ============================================================
            function val = localRmse(a, b)
                a = a(:);
                b = b(:);
                val = sqrt(mean((a - b).^2));
            end

            function [xhat, P] = localKf1d(y, Q, R, x0, P0)
                T = numel(y);
                xhat = zeros(1, T);
                P = zeros(1, T);

                x = x0;
                Pk = P0;

                for t = 1:T
                    % Predict
                    Pk = Pk + Q;

                    % Update
                    K = Pk / (Pk + R);
                    x = x + K * (y(t) - x);
                    Pk = (1 - K) * Pk;

                    xhat(t) = x;
                    P(t) = Pk;
                end
            end
        end
    end
end