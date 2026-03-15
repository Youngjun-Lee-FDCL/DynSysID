classdef VGPSSM < handle
    properties
        % Data
        Y
        U
        T
        numObs
        numInputs

        % Components
        options
        gp
        obs
        qU
        stateInferencer

        % Fixed in Phase 1
        thetaQ
        px0

        % Diagnostics
        history
        isInitialised logical = false
        diagnostics
    end

    methods
        function obj = VGPSSM(options)
            if nargin < 1 || isempty(options)
                obj.options = VGPSSMOptions();
            else
                obj.options = options;
            end
        end

        function initialize(obj, Y, U)
            if nargin < 3 || isempty(U)
                U = zeros(size(Y,1), 0);
            end

            obj.Y = Y;
            obj.U = U;
            obj.T = size(Y,1);
            obj.numObs = size(Y,2);
            obj.numInputs = size(U,2);

            Dx = obj.options.numStates;
            M  = obj.options.numInducingPoints;

            %-----------------------------
            % GP initialisation
            %-----------------------------
            covfunc  = obj.options.covfunc;
            meanfunc = obj.options.meanfunc;

            if isempty(obj.options.gpHypCov)
                hyp = GPMLDynamics.makeDefaultHyp(Dx + obj.numInputs);
            else
                hyp = struct();
                hyp.cov = obj.options.gpHypCov;
                hyp.mean = [];
            end

            % Inducing inputs Z: simple initialisation
            Zx = 3 * randn(M, Dx);

            if obj.numInputs > 0
                idx = randperm(size(U,1), M);
                Zu = U(idx, :);
            else
                Zu = zeros(M, 0);
            end

            Z = [Zx, Zu];

            obj.gp = GPMLDynamics(covfunc, meanfunc, hyp, Z, Dx);

            %-----------------------------
            % Observation model
            %-----------------------------
            if Dx >= obj.numObs
                C0 = [eye(obj.numObs), zeros(obj.numObs, Dx - obj.numObs)];
            else
                C0 = eye(obj.numObs, Dx);
            end
            R0 = eye(obj.numObs);

            obj.obs = LinearGaussianObs(C0, R0);

            %-----------------------------
            % Transition noise Q
            %-----------------------------
            obj.thetaQ = eye(Dx);

            %-----------------------------
            % Variational posterior q(u)
            %-----------------------------
            mu0 = obj.initialiseQuMean();
            Sigma0 = blkdiagN(eye(M), Dx);

            obj.qU = VariationalGaussian(Dx, M);
            obj.qU.setFromMoments(mu0, Sigma0);

            %-----------------------------
            % State posterior inferencer
            %-----------------------------
            obj.stateInferencer = SMCInferencer( ...
                obj.options.numParticles, ...
                Dx, ...
                obj.options.fixedLag);

            %-----------------------------
            % History
            %-----------------------------
            obj.diagnostics = VGPSSMDiagnostics();
            obj.history = obj.diagnostics.history;

            obj.isInitialised = true;
        end

        function fit(obj, Y, U)
            if ~obj.isInitialised
                obj.initialize(Y, U);
            end

            for iter = 1:obj.options.maxIter
                if obj.options.verbose
                    fprintf('Iter %d / %d\n', iter, obj.options.maxIter);
                end
                obj.runIteration(iter);
            end

            if obj.options.verbose
                obj.diagnostics.plotTrainingCurves();
                obj.diagnostics.plotLatestMiniBatchFit(obj);
            end
        end

        function runIteration(obj, iter)
            [tt, numMiniBatches] = sampleMiniBatch( ...
                obj.T, ...
                obj.options.miniBatchLength, ...
                obj.options.localSmoothingBuffer);

            qx = obj.stateInferencer.sampleTrajectoryPosterior(obj, obj.Y, obj.U, tt);

            rho = obj.computeRho(iter);

            % better name: scaleFactor = obj.T / numel(tt);
            scaleFactor = numMiniBatches;
            obj.updateQu(qx, rho, scaleFactor);

            gamma = obj.computeGamma(iter);
            if gamma > 0
                Ybatch = obj.Y(tt, :);
                gradC = obj.obs.gradCFromParticles(Ybatch, qx.Xt);
                obj.obs.updateC(gradC, gamma);
            end

            obj.diagnostics.update(obj, iter, rho, gamma, qx);
            obj.history = obj.diagnostics.history;

            if obj.options.verbose
                obj.diagnostics.printLast();
            end
        end

        function rho = computeRho(obj, iter)
            if iter < 20
                rho = obj.options.rhoInit;
            else
                rho = obj.options.rhoFinalScale * ...
                    (obj.options.maxIter - iter) / obj.options.maxIter;
                rho = max(rho, 0.05);
            end
        end

        function gamma = computeGamma(obj, iter)
            if iter < obj.options.cUpdateStartIter
                gamma = 0;
            else
                gamma = obj.options.gammaC * ...
                    (obj.options.maxIter - iter) / obj.options.maxIter;
                gamma = max(gamma, 0);
            end
        end

        function [mAux, extraLogWeight] = getAuxiliaryTransitionInfo(obj, XU)
            % Auxiliary model of Eq. (12)
            %
            % transition: x_{t+1} ~ N(A(x_t,u_t) * mu_u, Q)
            % extra weight: -1/2 tr(Q^{-1}(B + A Sigma A^T))

            Dx = obj.options.numStates;
            N  = size(XU,1);
            M  = obj.options.numInducingPoints;

            % ----- q(u) moments -----
            [muUmat, SigmaBlocks] = obj.getQuMomentsByOutput();
            % muUmat     : [M x Dx]
            % SigmaBlocks: cell{Dx}, each [M x M]

            % ----- GP conditional terms -----
            A = obj.gp.computeA(XU);      % [N x M]
            B = obj.gp.computeB(XU);      % [N x Dx]
            m = obj.gp.mean(XU);          % [N x Dx]

            mAux = zeros(N, Dx);
            extraLogWeight = zeros(N,1);

            Qinv = inv(obj.thetaQ);

            for n = 1:N
                An = A(n,:);   % [1 x M]

                ASAt_diag = zeros(1, Dx);
                for d = 1:Dx
                    mAux(n,d) = m(n,d) + An * muUmat(:,d);

                    Sigmad = SigmaBlocks{d};
                    ASAt_diag(d) = An * Sigmad * An';
                end

                % Assuming independent outputs in B(n,:) representation
                Sdiag = B(n,:) + ASAt_diag;

                if isdiag(obj.thetaQ)
                    extraLogWeight(n) = -0.5 * sum(diag(Qinv)' .* Sdiag);
                else
                    % If Q is full and outputs are treated independently,
                    % use diagonal approximation here
                    extraLogWeight(n) = -0.5 * trace(Qinv * diag(Sdiag));
                end
            end
        end

        function logp = logAuxiliaryTransitionDensity(obj, XU, Xnext)
            % log psi(x_{t+1}|x_t) for backward simulation
            %
            % psi = exp(extraLogWeight(x_t)) * N(x_{t+1} | mAux(x_t), Q)

            [mAux, extraLogWeight] = obj.getAuxiliaryTransitionInfo(XU);

            if isdiag(obj.thetaQ)
                qdiag = diag(obj.thetaQ)';
                logGauss = SMCInferencer.gaussianDiagLogPdfRows( ...
                    Xnext, mAux, repmat(qdiag, size(Xnext,1), 1));
            else
                logGauss = SMCInferencer.gaussianFullLogPdfRows(Xnext, mAux, obj.thetaQ);
            end

            logp = extraLogWeight + logGauss;
        end

        function [muUmat, SigmaBlocks] = getQuMomentsByOutput(obj)
            % Convert q(u) moments into output-wise blocks
            %
            % Expected storage:
            %   qU mean vector : [M*Dx x 1]
            %   qU covariance  : [M*Dx x M*Dx]

            muVec = obj.qU.mu;
            Sigma = obj.qU.Sigma;

            Dx = obj.options.numStates;
            M  = obj.options.numInducingPoints;

            muUmat = zeros(M, Dx);
            SigmaBlocks = cell(Dx,1);

            for d = 1:Dx
                ii = (d-1)*M + (1:M);
                muUmat(:,d) = muVec(ii);
                SigmaBlocks{d} = Sigma(ii,ii);
            end
        end

        function updateQu(obj, qx, rho, numMiniBatches)
            %UPDATEQU Update q(u) using Eq. (11) / Eq. (14)-style natural parameters
            %
            % qx.Xt   : [Lb x Dx x Np]   sampled x_t
            % qx.Xtp1 : [Lb x Dx x Np]   sampled x_{t+1}
            % qx.tt   : [Lb x 1]         time indices for x_t
            %
            % Assumptions in this implementation:
            %   1) independent GP output dimensions
            %   2) diagonal Q for Phase-1
            %   3) q(u) stored as concatenated blocks [u^(1); ...; u^(Dx)]
            %
            % Natural parameters:
            %   eta1_d = Q_d^{-1} * sum_t E[ A_t' (x_{t+1,d} - m_d(x_t,u_t)) ]
            %   eta2_d = -1/2 * ( Kzz^{-1} + Q_d^{-1} * sum_t E[ A_t' A_t ] )

            Dx = obj.options.numStates;
            M  = obj.options.numInducingPoints;

            Lb = size(qx.Xt, 1);      % mini-batch segment length S
            Np = size(qx.Xt, 3);      % number of trajectory samples / particles

            if ~isdiag(obj.thetaQ)
                error('updateQu: current implementation assumes diagonal Q.');
            end

            qdiag = diag(obj.thetaQ);         % [Dx x 1]
            invQdiag = 1 ./ max(qdiag, 1e-12);

            % SVI scaling: for a sampled segment of length S, scale by T/S.
            % If caller passes numMiniBatches = T/S, this is already correct.
            scaleFactor = numMiniBatches;

            % Prior precision block
            Kzz = obj.gp.Kzz();
            KzzInv = obj.gp.solveKzz(eye(M));

            eta1New = zeros(M*Dx, 1);
            eta2New = zeros(M*Dx, M*Dx);

            % Start eta2 from prior term: -1/2 * Kzz^{-1} for each output block
            for d = 1:Dx
                ii = (d-1)*M + (1:M);
                eta2New(ii,ii) = -0.5 * KzzInv;
            end

            % Accumulate sufficient statistics over mini-batch
            for i = 1:Lb
                t = qx.tt(i);

                Xt_i   = squeeze(qx.Xt(i,:,:))';    % [Np x Dx]
                Xtp1_i = squeeze(qx.Xtp1(i,:,:))';  % [Np x Dx]

                if size(Xt_i,1) ~= Np
                    Xt_i = Xt_i';
                end
                if size(Xtp1_i,1) ~= Np
                    Xtp1_i = Xtp1_i';
                end

                ut = obj.U(t, :);
                XU = [Xt_i, repmat(ut, Np, 1)];     % [Np x (Dx+Du)]

                % A_t(x_t,u_t): [Np x M]
                A = obj.gp.computeA(XU);

                % mean function m(x_t,u_t): [Np x Dx]
                m = obj.gp.mean(XU);

                % residual to which sparse GP contributes
                R = Xtp1_i - m;                     % [Np x Dx]

                % Monte Carlo estimates of expectations under q(x_t,x_{t+1})
                EAtr = (A' * R) / Np;               % [M x Dx], = E[A_t' (x_{t+1}-m_t)]
                EAtA = (A' * A) / Np;               % [M x M], = E[A_t' A_t]

                for d = 1:Dx
                    ii = (d-1)*M + (1:M);

                    % Eq. (11)/(14): eta1 contribution
                    eta1New(ii) = eta1New(ii) + scaleFactor * invQdiag(d) * EAtr(:,d);

                    % Eq. (11)/(14): eta2 contribution
                    eta2New(ii,ii) = eta2New(ii,ii) ...
                        - 0.5 * scaleFactor * invQdiag(d) * EAtA;
                end
            end

            % Damped update in natural-parameter space
            obj.qU.dampedUpdate(eta1New, eta2New, rho);
        end

        function mu0 = initialiseQuMean(obj)
            % Phase-1 robust default:
            % initialise around GP mean at Z
            mZ = obj.gp.mean(obj.gp.Z);   % [M x Dx]
            mu0 = reshape(mZ, [], 1);
        end

        function [mf, vf] = predictDynamics(obj, XU)
            [mf, vf] = obj.gp.predictFromQu(XU, obj.qU);
        end

        function [mf, vf] = predictNext(obj, XU)
            %PREDICTNEXT One-step dynamics prediction
            %
            % Input
            %   XU : [N x (Dx+Du)] or [1 x (Dx+Du)]
            %
            % Output
            %   mf : [N x Dx] posterior mean of next state
            %   vf : [N x Dx] posterior variance of next state (without Q added)

            [mf, vf] = obj.predictDynamics(XU);
        end

        function Yhat = predictObservation(obj, X)
            %PREDICTOBSERVATION Predict observation mean from state
            %
            % Input
            %   X    : [N x Dx]
            %
            % Output
            %   Yhat : [N x Dy]

            Yhat = obj.obs.predict(X);
        end

        function result = rollout(obj, x0, Ufuture, horizon, mode)
            %ROLLOUT Multi-step state rollout using learned dynamics
            %
            % Inputs
            %   x0      : [1 x Dx] initial state
            %   Ufuture : [H x Du] future inputs
            %   horizon : scalar, number of rollout steps
            %   mode    : 'mean' or 'sample'
            %
            % Output
            %   result struct with fields:
            %       .X        [H+1 x Dx] rolled-out states including x0
            %       .meanNext [H x Dx] one-step predictive means
            %       .varNext  [H x Dx] one-step predictive variances
            %       .mode
            %
            % Notes
            %   In 'mean' mode:
            %       x_{k+1} = mf
            %   In 'sample' mode:
            %       x_{k+1} ~ N(mf, vf + diag(Q))

            if nargin < 5 || isempty(mode)
                mode = 'mean';
            end

            Dx = obj.options.numStates;
            Du = obj.numInputs;

            if size(x0,1) ~= 1
                error('rollout: x0 must be [1 x Dx].');
            end
            if size(x0,2) ~= Dx
                error('rollout: x0 dimension mismatch.');
            end
            if size(Ufuture,1) < horizon
                error('rollout: Ufuture must have at least horizon rows.');
            end
            if size(Ufuture,2) ~= Du
                error('rollout: Ufuture input dimension mismatch.');
            end

            X = zeros(horizon+1, Dx);
            meanNext = zeros(horizon, Dx);
            varNext  = zeros(horizon, Dx);

            X(1,:) = x0;

            qdiag = diag(obj.thetaQ)';

            for k = 1:horizon
                xk = X(k,:);
                uk = Ufuture(k,:);
                XU = [xk, uk];

                [mf, vf] = obj.predictNext(XU);
                mf = reshape(mf, 1, Dx);
                vf = reshape(vf, 1, Dx);

                meanNext(k,:) = mf;
                varNext(k,:)  = vf;

                switch lower(mode)
                    case 'mean'
                        X(k+1,:) = mf;

                    case 'sample'
                        totalVar = max(vf + qdiag, 1e-9);
                        X(k+1,:) = mf + sqrt(totalVar) .* randn(1, Dx);

                    otherwise
                        error('rollout: mode must be ''mean'' or ''sample''.');
                end
            end

            result = struct();
            result.X = X;
            result.meanNext = meanNext;
            result.varNext = varNext;
            result.mode = mode;
        end

        function result = simulateObserved(obj, x0, Ufuture, horizon, mode)
            %SIMULATEOBSERVED Rollout state and predicted observation
            %
            % Output
            %   result.X  : [H+1 x Dx]
            %   result.Y  : [H+1 x Dy]
            %   result.meanNext
            %   result.varNext

            result = obj.rollout(x0, Ufuture, horizon, mode);
            result.Y = obj.predictObservation(result.X);
        end

        function result = rolloutFromTrainingIndex(obj, t0, horizon, mode)
            %ROLLOUTFROMTRAININGINDEX Roll out from state estimated from training data
            %
            % Phase-1 approximation:
            %   use first Dx observation channels as proxy for state.
            %
            % Inputs
            %   t0      : starting time index
            %   horizon : rollout length
            %   mode    : 'mean' or 'sample'

            if nargin < 4 || isempty(mode)
                mode = 'mean';
            end

            Dx = obj.options.numStates;

            if t0 < 1 || t0 > obj.T
                error('rolloutFromTrainingIndex: t0 out of range.');
            end
            if t0 + horizon - 1 > size(obj.U,1)
                error('rolloutFromTrainingIndex: future input range exceeds available U.');
            end
            if size(obj.Y,2) < Dx
                error('rolloutFromTrainingIndex: observation dimension < numStates.');
            end

            x0 = obj.Y(t0, 1:Dx);
            Ufuture = obj.U(t0:t0+horizon-1, :);

            result = obj.simulateObserved(x0, Ufuture, horizon, mode);
            result.t0 = t0;
        end

        function plotRolloutComparison(obj, t0, horizon, mode)
            %PLOTROLLOUTCOMPARISON Compare rollout with training observations
            %
            % Uses observation proxy state initialisation (Phase-1 style).

            if nargin < 4 || isempty(mode)
                mode = 'mean';
            end

            result = obj.rolloutFromTrainingIndex(t0, horizon, mode);

            tt = (t0:t0+horizon)';
            Ytrue = obj.Y(tt, :);
            Ypred = result.Y;

            Dy = size(Ytrue,2);

            figure;
            for j = 1:Dy
                subplot(Dy,1,j);
                plot(tt, Ytrue(:,j), 'k-', 'LineWidth', 1.2); hold on;
                plot(tt, Ypred(:,j), 'r--', 'LineWidth', 1.2);
                grid on;
                xlabel('t');
                ylabel(sprintf('y_%d', j));
                legend('True', 'Rollout Pred');
            end
        end

        function result = rolloutEnsemble(obj, x0, Ufuture, horizon, numEnsemble)
            %ROLLOUTENSEMBLE Multi-sample rollout ensemble
            %
            % Inputs
            %   x0          : [1 x Dx]
            %   Ufuture     : [H x Du]
            %   horizon     : scalar
            %   numEnsemble : number of sampled trajectories
            %
            % Output
            %   result struct:
            %       .Xens      [H+1 x Dx x Ne]
            %       .Yens      [H+1 x Dy x Ne]
            %       .Xmean     [H+1 x Dx]
            %       .Xstd      [H+1 x Dx]
            %       .Ymean     [H+1 x Dy]
            %       .Ystd      [H+1 x Dy]
            %       .numEnsemble
            %       .horizon

            if nargin < 5 || isempty(numEnsemble)
                numEnsemble = 50;
            end

            Dx = obj.options.numStates;
            Dy = obj.numObs;
            Ne = numEnsemble;

            Xens = zeros(horizon+1, Dx, Ne);
            Yens = zeros(horizon+1, Dy, Ne);

            for m = 1:Ne
                tmp = obj.simulateObserved(x0, Ufuture, horizon, 'sample');
                Xens(:,:,m) = tmp.X;
                Yens(:,:,m) = tmp.Y;
            end

            Xmean = mean(Xens, 3);
            Ymean = mean(Yens, 3);

            if Ne > 1
                Xstd = std(Xens, 0, 3);
                Ystd = std(Yens, 0, 3);
            else
                Xstd = zeros(horizon+1, Dx);
                Ystd = zeros(horizon+1, Dy);
            end

            result = struct();
            result.Xens = Xens;
            result.Yens = Yens;
            result.Xmean = Xmean;
            result.Xstd = Xstd;
            result.Ymean = Ymean;
            result.Ystd = Ystd;
            result.numEnsemble = Ne;
            result.horizon = horizon;
        end

        function result = rolloutEnsembleFromTrainingIndex(obj, t0, horizon, numEnsemble)
            %ROLLOUTENSEMBLEFROMTRAININGINDEX Ensemble rollout from a training index
            %
            % Phase-1 approximation:
            %   x0 is taken from first Dx observation channels at time t0.

            Dx = obj.options.numStates;

            if t0 < 1 || t0 > obj.T
                error('rolloutEnsembleFromTrainingIndex: t0 out of range.');
            end
            if t0 + horizon - 1 > size(obj.U,1)
                error('rolloutEnsembleFromTrainingIndex: future input range exceeds available U.');
            end
            if size(obj.Y,2) < Dx
                error('rolloutEnsembleFromTrainingIndex: observation dimension < numStates.');
            end

            x0 = obj.Y(t0, 1:Dx);
            Ufuture = obj.U(t0:t0+horizon-1, :);

            result = obj.rolloutEnsemble(x0, Ufuture, horizon, numEnsemble);
            result.t0 = t0;
        end

        function plotRolloutUncertainty(obj, t0, horizon, numEnsemble)
            %PLOTROLLOUTUNCERTAINTY Plot observed trajectory and ensemble uncertainty band
            %
            % Uses Ymean ± 2*Ystd as uncertainty band.

            if nargin < 4 || isempty(numEnsemble)
                numEnsemble = 50;
            end

            result = obj.rolloutEnsembleFromTrainingIndex(t0, horizon, numEnsemble);

            tt = (t0:t0+horizon)';
            Ytrue = obj.Y(tt, :);
            Ymean = result.Ymean;
            Ystd  = result.Ystd;

            Dy = size(Ytrue,2);

            figure;
            for j = 1:Dy
                subplot(Dy,1,j);

                lower = Ymean(:,j) - 2*Ystd(:,j);
                upper = Ymean(:,j) + 2*Ystd(:,j);

                fill([tt; flipud(tt)], [lower; flipud(upper)], ...
                    [0.85 0.85 1.0], 'EdgeColor', 'none'); hold on;
                plot(tt, Ymean(:,j), 'b--', 'LineWidth', 1.5);
                plot(tt, Ytrue(:,j), 'k-', 'LineWidth', 1.2);

                grid on;
                xlabel('t');
                ylabel(sprintf('y_%d', j));
                legend('Uncertainty band', 'Ensemble mean', 'True');
                title(sprintf('Rollout uncertainty for y_%d', j));
            end
        end

        function plotStateRolloutUncertainty(obj, t0, horizon, numEnsemble)
            %PLOTSTATEROLLOUTUNCERTAINTY Plot state ensemble mean and uncertainty band

            if nargin < 4 || isempty(numEnsemble)
                numEnsemble = 50;
            end

            result = obj.rolloutEnsembleFromTrainingIndex(t0, horizon, numEnsemble);

            tt = (t0:t0+horizon)';
            Xmean = result.Xmean;
            Xstd  = result.Xstd;

            Dx = size(Xmean,2);

            figure;
            for j = 1:Dx
                subplot(Dx,1,j);

                lower = Xmean(:,j) - 2*Xstd(:,j);
                upper = Xmean(:,j) + 2*Xstd(:,j);

                fill([tt; flipud(tt)], [lower; flipud(upper)], ...
                    [0.90 1.0 0.90], 'EdgeColor', 'none'); hold on;
                plot(tt, Xmean(:,j), 'g--', 'LineWidth', 1.5);

                grid on;
                xlabel('t');
                ylabel(sprintf('x_%d', j));
                legend('Uncertainty band', 'Ensemble mean');
                title(sprintf('State rollout uncertainty for x_%d', j));
            end
        end
    end


    methods (Static)
        function test()
            clear; clc;

            % GPML path must already be added
            % addpath(genpath('C:\gpml-matlab-v4.2-2018-06-11'));

            T  = 300;
            Dx = 2;
            Du = 1;
            Dy = 2;

            t = (0:T-1)' * 0.1;
            U = sin(0.3*t);

            Xtrue = zeros(T, Dx);
            for k = 1:T-1
                Xtrue(k+1,1) = Xtrue(k,1) + 0.1*Xtrue(k,2);
                Xtrue(k+1,2) = 0.95*Xtrue(k,2) + 0.2*sin(Xtrue(k,1)) + 0.1*U(k);
            end

            Y = Xtrue + 0.05*randn(T, Dy);

            opts = VGPSSMOptions( ...
                'numStates', Dx, ...
                'numInducingPoints', 15, ...
                'numParticles', 50, ...
                'maxIter', 20, ...
                'miniBatchLength', 50, ...
                'verbose', true);

            opts.covfunc  = @covSEard;
            opts.meanfunc = GPMLDynamics.makeStateIdentityMean(Dx);
            opts.gpHypCov = [zeros(Dx+Du,1); log(1.0)];

            model = VGPSSM(opts);
            model.fit(Y, U);

            disp(model.obs.C)

            % Assume model is already trained
            t0 = 50;
            H  = 30;

            resMean = model.rolloutFromTrainingIndex(t0, H, 'mean');
            resSamp = model.rolloutFromTrainingIndex(t0, H, 'sample');

            disp(size(resMean.X))   % [H+1 x Dx]
            disp(size(resMean.Y))   % [H+1 x Dy]

            model.plotRolloutComparison(t0, H, 'mean');
        end
    end
end