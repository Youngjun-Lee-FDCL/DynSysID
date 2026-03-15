classdef SMCInferencer < handle
    properties
        numParticles (1,1) double = 100
        numStates    (1,1) double = 2
        resampleThreshold (1,1) double = 0.5
        initStd (1,1) double = 0.1
        fixedLag (1,1) double = 5
    end

    methods
        function obj = SMCInferencer(numParticles, numStates, fixedLag)
            obj.numParticles = numParticles;
            obj.numStates = numStates;
            if nargin >= 3
                obj.fixedLag = fixedLag;
            end
        end

        function qx = sampleTrajectoryPosterior(obj, model, Y, U, tt)
            % Sample from auxiliary smoothing model of Eq. (12)
            %
            % q*(x) ∝ p(x0) prod_t p(y_t|x_t)
            %         exp(extraTerm(x_{t-1})) N(x_t | A_{t-1} mu, Q)

            pf = obj.runAuxiliaryBootstrapFilter(model, Y, U);

            L  = numel(tt);
            Dx = obj.numStates;
            Np = obj.numParticles;
            T  = size(Y,1);

            Xt   = zeros(L, Dx, Np);
            Xtp1 = zeros(L, Dx, Np);
            W    = ones(L, Np) / Np;

            for i = 1:L
                t = tt(i);

                if t+1 > T
                    error('SMCInferencer: need t+1 <= T.');
                end

                tau = min(t + obj.fixedLag, T);

                for p = 1:Np
                    xPath = obj.backwardSimulateAuxPath(model, pf, U, t, tau);

                    Xt(i,:,p) = xPath(1,:);

                    if size(xPath,1) >= 2
                        Xtp1(i,:,p) = xPath(2,:);
                    else
                        Xtp1(i,:,p) = xPath(1,:);
                    end
                end
            end

            qx = struct();
            qx.tt   = tt;
            qx.Xt   = Xt;
            qx.Xtp1 = Xtp1;
            qx.W    = W;
        end

        function xPath = backwardSimulateAuxPath(obj, model, pf, U, t0, tau)
            % Backward simulation using auxiliary transition density
            %
            % p(x_k | x_{k+1}, y_{1:k}) ∝ w_k^i * psi(x_{k+1}|x_k^i)
            % psi(x_{k+1}|x_k) = exp(extraLogWeight(x_k)) * N(x_{k+1}|m_k,Q)

            Dx = obj.numStates;
            Np = obj.numParticles;

            wtau = pf.weights(tau,:)';
            wtau = wtau / sum(wtau);
            idxNext = obj.sampleDiscrete(wtau);

            xNext = reshape(pf.particles(tau, idxNext, :), 1, Dx);

            pathRev = zeros(tau - t0 + 1, Dx);
            pathRev(1,:) = xNext;
            pos = 2;

            for k = tau-1:-1:t0
                Xk = squeeze(pf.particles(k,:,:));   % [Np x Dx]
                if size(Xk,1) ~= Np
                    Xk = Xk';
                end

                wk = pf.weights(k,:)';
                wk = wk / sum(wk);

                uk = U(k,:);
                XU = [Xk, repmat(uk, Np, 1)];

                logPsi = model.logAuxiliaryTransitionDensity(XU, repmat(xNext, Np, 1));
                logBw  = log(wk + 1e-300) + logPsi;

                bw = obj.normaliseLogWeights(logBw);
                idxk = obj.sampleDiscrete(bw);

                xk = Xk(idxk,:);
                pathRev(pos,:) = xk;

                xNext = xk;
                pos = pos + 1;
            end

            xPath = flipud(pathRev);
        end

        function pf = runAuxiliaryBootstrapFilter(obj, model, Y, U)
            % Bootstrap PF for the auxiliary model of Eq. (12)

            T  = size(Y,1);
            Dx = obj.numStates;
            Np = obj.numParticles;

            particles = zeros(T, Np, Dx);
            weights   = zeros(T, Np);
            ancestors = zeros(T, Np);

            % Initial particles
            x0mean = obj.initialStateMean(Y, Dx);
            X0 = repmat(x0mean, Np, 1) + obj.initStd * randn(Np, Dx);

            % t = 1 weight: p(y1 | x1) structure is handled after propagation,
            % but for bootstrap initialisation we use observation at time 1
            logw0 = model.obs.particleLogLikelihood(Y(1,:), X0);
            w0 = obj.normaliseLogWeights(logw0);

            particles(1,:,:) = X0;
            weights(1,:) = w0;
            ancestors(1,:) = 1:Np;

            for t = 1:T-1
                wt = weights(t,:)';
                ess = 1 / sum(wt.^2);

                Xprev = squeeze(particles(t,:,:));   % [Np x Dx]
                if size(Xprev,1) ~= Np
                    Xprev = Xprev';
                end

                if ess < obj.resampleThreshold * Np
                    aidx = obj.systematicResample(wt);
                    Xres = Xprev(aidx,:);
                    wres = ones(Np,1) / Np;
                else
                    aidx = (1:Np)';
                    Xres = Xprev;
                    wres = wt;
                end

                ut = U(t,:);
                XU = [Xres, repmat(ut, Np, 1)];

                % Auxiliary transition:
                % x_{t+1} ~ N(A_t mu, Q)
                [mAux, extraLogWeight] = model.getAuxiliaryTransitionInfo(XU);

                Xnext = obj.sampleFromTransitionMeanQ(mAux, model.thetaQ);

                % Weight:
                % w_{t+1} ∝ previousWeight * exp(extraTerm(x_t)) * p(y_{t+1}|x_{t+1})
                logwObs = model.obs.particleLogLikelihood(Y(t+1,:), Xnext);
                logw = log(wres + 1e-300) + extraLogWeight + logwObs;
                wnext = obj.normaliseLogWeights(logw);

                particles(t+1,:,:) = Xnext;
                weights(t+1,:)     = wnext;
                ancestors(t+1,:)   = aidx;
            end

            pf = struct();
            pf.particles = particles;
            pf.weights   = weights;
            pf.ancestors = ancestors;
        end

        function Xnext = sampleFromTransitionMeanQ(obj, meanX, Q)
            % Sample row-wise from N(meanX, Q)
            [N, Dx] = size(meanX);

            if isvector(Q)
                qdiag = Q(:)';
                Xnext = meanX + randn(N, Dx) .* sqrt(repmat(qdiag, N, 1));
                return;
            end

            if isdiag(Q)
                qdiag = diag(Q)';
                Xnext = meanX + randn(N, Dx) .* sqrt(repmat(qdiag, N, 1));
            else
                % full covariance
                Lq = chol(Q + 1e-9*eye(Dx), 'lower');
                Xnext = meanX + randn(N, Dx) * Lq';
            end
        end

        function x0mean = initialStateMean(obj, Y, Dx)
            if size(Y,2) >= Dx
                x0mean = Y(1,1:Dx);
            else
                x0mean = zeros(1,Dx);
            end
        end
    end

    methods (Static)
        function logp = gaussianDiagLogPdfRows(X, Mu, Var)
            diff2 = (X - Mu).^2;
            logp = -0.5 * sum(log(2*pi*Var) + diff2 ./ Var, 2);

            bad = ~isfinite(logp);
            if any(bad)
                logp(bad) = -1e12;
            end
        end

        function logp = gaussianFullLogPdfRows(X, Mu, Sigma)
            % Row-wise full Gaussian log-pdf with common covariance Sigma
            [N, D] = size(X);
            L = chol(Sigma + 1e-9*eye(D), 'lower');
            XC = (X - Mu) / L';
            quad = sum(XC.^2, 2);
            logdet = 2 * sum(log(diag(L)));
            logp = -0.5 * (D*log(2*pi) + logdet + quad);
        end

        function w = normaliseLogWeights(logw)
            c = max(logw);
            w = exp(logw - c);
            s = sum(w);
            if s <= 0 || ~isfinite(s)
                w = ones(size(logw)) / numel(logw);
            else
                w = w / s;
            end
        end

        function idx = systematicResample(w)
            N = numel(w);
            edges = min([0; cumsum(w(:))], 1);
            edges(end) = 1;

            u1 = rand / N;
            u = u1 + (0:N-1)' / N;

            idx = zeros(N,1);
            j = 1;
            for i = 1:N
                while u(i) > edges(j+1)
                    j = j + 1;
                end
                idx(i) = j;
            end
        end

        function k = sampleDiscrete(w)
            c = cumsum(w(:));
            r = rand;
            k = find(r <= c, 1, 'first');
            if isempty(k)
                k = numel(w);
            end
        end
    end
end