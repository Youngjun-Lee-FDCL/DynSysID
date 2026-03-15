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

        % Learned model parameters
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

            covfunc  = obj.options.covfunc;
            meanfunc = obj.options.meanfunc;

            if isempty(obj.options.gpHypCov)
                hyp = GPMLDynamics.makeDefaultHyp(Dx + obj.numInputs);
            else
                hyp = struct();
                hyp.cov = obj.options.gpHypCov(:);
                hyp.mean = [];
            end

            Z = obj.initialiseInducingInputs(M, Dx);
            obj.gp = GPMLDynamics(covfunc, meanfunc, hyp, Z, Dx);
            obj.gp.jitter = obj.options.jitter;

            if Dx >= obj.numObs
                C0 = [eye(obj.numObs), zeros(obj.numObs, Dx - obj.numObs)];
            else
                C0 = eye(obj.numObs, Dx);
            end
            yStd = std(Y, 0, 1);
            yStd(~isfinite(yStd) | yStd < 1e-8) = 1.0;
            R0 = diag(max(0.05 * yStd(:).^2, obj.options.minObsVar));
            obj.obs = LinearGaussianObs(C0, R0);

            obj.thetaQ = diag(max(0.10 * ones(Dx,1), obj.options.minProcessVar));
            obj.initialisePx0();

            mu0 = obj.initialiseQuMean();
            Sigma0 = blkdiagN(obj.gp.Kzz(), Dx);
            obj.qU = VariationalGaussian(Dx, M);
            obj.qU.setFromMoments(mu0, Sigma0);

            obj.stateInferencer = SMCInferencer(...
                obj.options.numParticles, Dx, obj.options.fixedLag);

            obj.diagnostics = VGPSSMDiagnostics();
            obj.history = obj.diagnostics.history;
            obj.isInitialised = true;
        end

        function Z = initialiseInducingInputs(obj, M, Dx)
            switch lower(obj.options.inducingStateInit)
                case 'data'
                    if obj.numObs >= Dx
                        Xproxy = obj.Y(:,1:Dx);
                    else
                        Xproxy = [obj.Y, zeros(obj.T, Dx - obj.numObs)];
                    end
                    idx = round(linspace(1, size(Xproxy,1), M));
                    Zx = Xproxy(idx,:);
                    if size(Zx,1) < M
                        Zx = [Zx; repmat(mean(Xproxy,1), M-size(Zx,1), 1)];
                    end
                otherwise
                    Zx = 3 * randn(M, Dx);
            end

            if obj.numInputs > 0
                idxu = round(linspace(1, size(obj.U,1), M));
                Zu = obj.U(idxu,:);
                if size(Zu,1) < M
                    Zu = [Zu; repmat(mean(obj.U,1), M-size(Zu,1), 1)];
                end
            else
                Zu = zeros(M, 0);
            end
            Z = [Zx, Zu];
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
            [tt, numMiniBatches] = sampleMiniBatch(...
                obj.T, obj.options.miniBatchLength, obj.options.localSmoothingBuffer);

            qx = obj.stateInferencer.sampleTrajectoryPosterior(obj, obj.Y, obj.U, tt);
            rho = obj.computeRho(iter);
            obj.updateQu(qx, rho, numMiniBatches);

            gammaObs = obj.computeGamma(iter, obj.options.obsUpdateStartIter, obj.options.gammaObs);
            if gammaObs > 0
                gradC = obj.obs.gradCFromParticles(obj.Y(tt,:), qx.Xt, qx.W);
                obj.obs.updateC(gradC, gammaObs);
            end

            obj.updateThetaParameters(qx, iter);
            obj.updateInducingInputs(qx, iter);

            obj.diagnostics.update(obj, iter, rho, gammaObs, qx);
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
                    (obj.options.maxIter - iter) / max(obj.options.maxIter, 1);
                rho = max(rho, 0.05);
            end
        end

        function gamma = computeGamma(obj, iter, startIter, gamma0)
            if iter < startIter
                gamma = 0;
            else
                gamma = gamma0 * (obj.options.maxIter - iter) / max(obj.options.maxIter, 1);
                gamma = max(gamma, 0);
            end
        end

        function initialisePx0(obj)
            Dx = obj.options.numStates;
            if obj.numObs >= Dx
                mu0 = obj.Y(1,1:Dx)';
            else
                mu0 = zeros(Dx,1);
            end
            P0 = obj.options.initStateVar * eye(Dx);
            obj.px0 = struct('mean', mu0, 'cov', P0);
        end

        function X0 = sampleInitialStatePrior(obj, N)
            mu0 = obj.px0.mean(:)';
            P0 = 0.5 * (obj.px0.cov + obj.px0.cov');
            P0 = P0 + obj.options.jitter * eye(size(P0));
            L0 = chol(P0, 'lower');
            X0 = repmat(mu0, N, 1) + randn(N, numel(mu0)) * L0';
        end

        function [mAux, extraLogWeight] = getAuxiliaryTransitionInfo(obj, XU)
            Dx = obj.options.numStates;
            N  = size(XU,1);

            [muUmat, SigmaBlocks] = obj.getQuMomentsByOutput();
            A  = obj.gp.computeA(XU);
            B  = obj.gp.computeB(XU);
            m  = obj.gp.mean(XU);
            mZ = obj.gp.mean(obj.gp.Z);

            mAux = zeros(N, Dx);
            extraLogWeight = zeros(N,1);

            qdiag = diag(obj.thetaQ)';
            invQdiag = 1 ./ max(qdiag, obj.options.minProcessVar);

            for n = 1:N
                An = A(n,:);
                ASAtDiag = zeros(1, Dx);
                for d = 1:Dx
                    mAux(n,d) = m(n,d) + An * (muUmat(:,d) - mZ(:,d));
                    ASAtDiag(d) = An * SigmaBlocks{d} * An';
                end
                Sdiag = B(n,:) + ASAtDiag;
                extraLogWeight(n) = -0.5 * sum(invQdiag .* Sdiag);
            end
        end

        function logp = logAuxiliaryTransitionDensity(obj, XU, Xnext)
            [mAux, extraLogWeight] = obj.getAuxiliaryTransitionInfo(XU);
            if isdiag(obj.thetaQ)
                qdiag = diag(obj.thetaQ)';
                logGauss = SMCInferencer.gaussianDiagLogPdfRows(...
                    Xnext, mAux, repmat(qdiag, size(Xnext,1), 1));
            else
                logGauss = SMCInferencer.gaussianFullLogPdfRows(Xnext, mAux, obj.thetaQ);
            end
            logp = extraLogWeight + logGauss;
        end

        function [muUmat, SigmaBlocks] = getQuMomentsByOutput(obj)
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
            Dx = obj.options.numStates;
            M  = obj.options.numInducingPoints;
            Lb = size(qx.Xt, 1);
            Np = size(qx.Xt, 3);

            if ~isdiag(obj.thetaQ)
                error('updateQu: current implementation assumes diagonal Q.');
            end

            qdiag = diag(obj.thetaQ);
            invQdiag = 1 ./ max(qdiag, obj.options.minProcessVar);
            Kzz = obj.gp.Kzz();
            KzzInv = obj.gp.solveKzz(eye(M));
            mZ = obj.gp.mean(obj.gp.Z);

            eta1New = zeros(M*Dx, 1);
            eta2New = zeros(M*Dx, M*Dx);

            for d = 1:Dx
                ii = (d-1)*M + (1:M);
                eta1New(ii) = KzzInv * mZ(:,d);
                eta2New(ii,ii) = -0.5 * KzzInv;
            end

            for i = 1:Lb
                t = qx.tt(i);
                Xt_i   = squeeze(qx.Xt(i,:,:))';
                Xtp1_i = squeeze(qx.Xtp1(i,:,:))';
                if size(Xt_i,1) ~= Np, Xt_i = Xt_i'; end
                if size(Xtp1_i,1) ~= Np, Xtp1_i = Xtp1_i'; end

                wt = reshape(qx.W(i,:), [], 1);
                wt = wt / max(sum(wt), eps);

                ut = obj.U(t,:);
                XU = [Xt_i, repmat(ut, Np, 1)];
                A  = obj.gp.computeA(XU);
                m  = obj.gp.mean(XU);
                AmZ = A * mZ;
                R = Xtp1_i - (m - AmZ);

                EAtr = A' * (R .* wt);
                EAtA = A' * (A .* wt);

                for d = 1:Dx
                    ii = (d-1)*M + (1:M);
                    eta1New(ii) = eta1New(ii) + numMiniBatches * invQdiag(d) * EAtr(:,d);
                    eta2New(ii,ii) = eta2New(ii,ii) - 0.5 * numMiniBatches * invQdiag(d) * EAtA;
                end
            end

            obj.qU.dampedUpdate(eta1New, eta2New, rho);
        end

        function mu0 = initialiseQuMean(obj)
            mZ = obj.gp.mean(obj.gp.Z);
            mu0 = reshape(mZ, [], 1);
        end

        function [mf, vf] = predictDynamics(obj, XU)
            [mf, vf] = obj.gp.predictFromQu(XU, obj.qU);
        end

        function [mf, vf] = predictNext(obj, XU)
            [mf, vf] = obj.predictDynamics(XU);
        end

        function Yhat = predictObservation(obj, X)
            Yhat = obj.obs.predict(X);
        end

        function updateThetaParameters(obj, qx, iter)
            gammaQ = obj.computeGamma(iter, obj.options.thetaUpdateStartIter, obj.options.gammaQ);
            gammaR = obj.computeGamma(iter, obj.options.thetaUpdateStartIter, obj.options.gammaR);
            gammaH = obj.computeGamma(iter, obj.options.thetaUpdateStartIter, obj.options.gammaHyp);

            if gammaQ > 0, obj.updateThetaQ(qx, gammaQ); end
            if gammaR > 0, obj.updateObservationNoise(qx, gammaR); end
            if gammaH > 0, obj.updateGpHypCov(qx, gammaH); end
        end

        function updateThetaQ(obj, qx, gammaQ)
            [residuals, ~] = obj.collectTransitionResiduals(qx);
            if isempty(residuals), return; end

            qNew = mean(residuals.^2, 1)';
            qNew = max(qNew, obj.options.minProcessVar);
            qOld = diag(obj.thetaQ);
            qUpd = (1 - gammaQ) * qOld + gammaQ * qNew;
            obj.thetaQ = diag(max(qUpd, obj.options.minProcessVar));
        end

        function updateObservationNoise(obj, qx, gammaR)
            [Ystack, Xstack] = obj.collectStateSamples(qx);
            if isempty(Xstack), return; end
            Yhat = Xstack * obj.obs.C';
            res = Ystack - Yhat;
            rNew = mean(res.^2, 1)';
            rOld = diag(obj.obs.R);
            rUpd = (1 - gammaR) * rOld + gammaR * max(rNew, obj.options.minObsVar);
            obj.obs.R = diag(max(rUpd, obj.options.minObsVar));
        end

        function updateGpHypCov(obj, qx, gammaHyp)
            hyp0 = obj.gp.hyp.cov(:);
            if isempty(hyp0), return; end

            epsFD = obj.options.hypFiniteDiffEps;
            grad = zeros(size(hyp0));
            base = hyp0;

            for k = 1:numel(base)
                hp = base; hm = base;
                hp(k) = hp(k) + epsFD;
                hm(k) = hm(k) - epsFD;
                obj.gp.hyp.cov = hp;
                Jp = obj.transitionSurrogateObjective(qx);
                obj.gp.hyp.cov = hm;
                Jm = obj.transitionSurrogateObjective(qx);
                grad(k) = (Jp - Jm) / (2 * epsFD);
            end

            obj.gp.hyp.cov = base - gammaHyp * grad;
        end

        function updateInducingInputs(obj, qx, iter)
            gammaZ = obj.computeGamma(iter, obj.options.zUpdateStartIter, obj.options.gammaZ);
            if gammaZ <= 0, return; end

            Dx = obj.options.numStates;
            M  = obj.options.numInducingPoints;
            Xmean = stableMeanTrajectory(qx.Xt);
            Umini = obj.U(qx.tt,:);
            XU = [Xmean, Umini];

            if isempty(XU)
                return;
            end

            idx = round(linspace(1, size(XU,1), min(M, size(XU,1))));
            Zcand = XU(idx,:);
            if size(Zcand,1) < M
                Zcand = [Zcand; repmat(mean(XU,1), M-size(Zcand,1), 1)];
            end

            Zold = obj.gp.Z;
            Znew = (1 - gammaZ) * Zold + gammaZ * Zcand(1:M,:);
            obj.gp.setInducingInputs(Znew);
        end

        function J = transitionSurrogateObjective(obj, qx)
            [residuals, predVar] = obj.collectTransitionResiduals(qx);
            if isempty(residuals)
                J = 0;
                return;
            end
            qdiag = diag(obj.thetaQ)';
            totalVar = max(predVar + repmat(qdiag, size(predVar,1), 1), 1e-9);
            J = 0.5 * sum(log(2*pi*totalVar) + residuals.^2 ./ totalVar, 'all');
            J = J / size(residuals,1);
        end

        function elbo = approximateELBO(obj, qx)
            [Ystack, Xstack] = obj.collectStateSamples(qx);
            obsTerm = 0;
            if ~isempty(Xstack)
                obsTerm = obj.obs.loglik(Ystack, Xstack) / size(Xstack,1);
            end
            dynTerm = -obj.transitionSurrogateObjective(qx);
            klTerm = obj.computeQuPriorKL();
            elbo = obsTerm + dynTerm - klTerm / max(numel(qx.tt),1);
        end

        function kl = computeQuPriorKL(obj)
            [mu, Sigma] = obj.qU.getMoments();
            Dx = obj.options.numStates;
            M  = obj.options.numInducingPoints;
            Kzz = obj.gp.Kzz();
            Kinv = obj.gp.solveKzz(eye(M));
            mZ = obj.gp.mean(obj.gp.Z);
            mu0 = reshape(mZ, [], 1);
            P0 = blkdiagN(Kzz, Dx);
            P0inv = blkdiagN(Kinv, Dx);

            diff = mu - mu0;
            n = numel(mu);
            [Ls, ps] = chol(Sigma + obj.options.jitter*eye(n), 'lower');
            [Lp, pp] = chol(P0 + obj.options.jitter*eye(n), 'lower');
            if ps ~= 0 || pp ~= 0
                kl = NaN;
                return;
            end
            logdetS = 2*sum(log(diag(Ls)));
            logdetP = 2*sum(log(diag(Lp)));
            kl = 0.5 * (trace(P0inv * Sigma) + diff' * P0inv * diff - n + logdetP - logdetS);
        end

        function [residuals, predVar] = collectTransitionResiduals(obj, qx)
            Dx = obj.options.numStates;
            Lb = size(qx.Xt, 1);
            Np = size(qx.Xt, 3);
            residuals = zeros(Lb * Np, Dx);
            predVar   = zeros(Lb * Np, Dx);
            row = 0;
            for i = 1:Lb
                t = qx.tt(i);
                Xt_i   = squeeze(qx.Xt(i,:,:))';
                Xtp1_i = squeeze(qx.Xtp1(i,:,:))';
                if size(Xt_i,1) ~= Np, Xt_i = Xt_i'; end
                if size(Xtp1_i,1) ~= Np, Xtp1_i = Xtp1_i'; end
                ut = obj.U(t,:);
                XU = [Xt_i, repmat(ut, Np, 1)];
                [mf, vf] = obj.predictDynamics(XU);
                ii = row + (1:Np);
                residuals(ii,:) = Xtp1_i - mf;
                predVar(ii,:) = vf;
                row = row + Np;
            end
        end

        function [Ystack, Xstack] = collectStateSamples(obj, qx)
            Lb = size(qx.Xt,1);
            Np = size(qx.Xt,3);
            Dx = obj.options.numStates;
            Dy = obj.numObs;
            Xstack = zeros(Lb * Np, Dx);
            Ystack = zeros(Lb * Np, Dy);
            row = 0;
            for i = 1:Lb
                Xt_i = squeeze(qx.Xt(i,:,:))';
                if size(Xt_i,1) ~= Np, Xt_i = Xt_i'; end
                ii = row + (1:Np);
                Xstack(ii,:) = Xt_i;
                Ystack(ii,:) = repmat(obj.Y(qx.tt(i),:), Np, 1);
                row = row + Np;
            end
        end

        function result = rollout(obj, x0, Ufuture, horizon, mode)
            if nargin < 5 || isempty(mode), mode = 'mean'; end
            Dx = obj.options.numStates;
            Du = obj.numInputs;
            if size(x0,1) ~= 1 || size(x0,2) ~= Dx
                error('rollout: x0 must be [1 x Dx].');
            end
            if size(Ufuture,1) < horizon || size(Ufuture,2) ~= Du
                error('rollout: Ufuture size mismatch.');
            end

            X = zeros(horizon+1, Dx);
            meanNext = zeros(horizon, Dx);
            varNext  = zeros(horizon, Dx);
            X(1,:) = x0;
            qdiag = diag(obj.thetaQ)';

            for k = 1:horizon
                XU = [X(k,:), Ufuture(k,:)];
                [mf, vf] = obj.predictNext(XU);
                mf = reshape(mf, 1, Dx);
                vf = reshape(vf, 1, Dx);
                meanNext(k,:) = mf;
                varNext(k,:) = vf;
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

            result = struct('X', X, 'meanNext', meanNext, 'varNext', varNext, 'mode', mode);
        end

        function result = simulateObserved(obj, x0, Ufuture, horizon, mode)
            result = obj.rollout(x0, Ufuture, horizon, mode);
            result.Y = obj.predictObservation(result.X);
        end

        function result = rolloutFromTrainingIndex(obj, t0, horizon, mode)
            if nargin < 4 || isempty(mode), mode = 'mean'; end
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
            if nargin < 4 || isempty(mode), mode = 'mean'; end
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
                grid on; xlabel('t'); ylabel(sprintf('y_%d', j));
                legend('True', 'Rollout Pred');
            end
        end
    end
end
