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
            T = size(Y,1);
            buf = model.options.localSmoothingBuffer;

            if model.options.useLocalWindow
                tStart = max(1, min(tt) - buf);
                tEnd   = min(T, max(tt) + max(obj.fixedLag, 1) + buf);
            else
                tStart = 1;
                tEnd   = T;
            end

            Ywin = Y(tStart:tEnd,:);
            Uwin = U(tStart:tEnd,:);
            pf = obj.runAuxiliaryBootstrapFilter(model, Ywin, Uwin, tStart);

            L  = numel(tt);
            Dx = obj.numStates;
            Np = obj.numParticles;
            Xt   = zeros(L, Dx, Np);
            Xtp1 = zeros(L, Dx, Np);
            W    = ones(L, Np) / Np;

            for p = 1:Np
                localPath = obj.backwardSimulateAuxPath(model, pf, Uwin, 1, size(Ywin,1));
                globalTimes = tStart:tEnd;
                for i = 1:L
                    t = tt(i);
                    iloc = t - tStart + 1;
                    if iloc < 1 || iloc > size(localPath,1)
                        error('SMCInferencer: local index out of range.');
                    end
                    Xt(i,:,p) = localPath(iloc,:);
                    ilocp1 = min(iloc + 1, size(localPath,1));
                    Xtp1(i,:,p) = localPath(ilocp1,:);
                end
            end

            qx = struct();
            qx.tt = tt;
            qx.Xt = Xt;
            qx.Xtp1 = Xtp1;
            qx.W = W;
            qx.window = [tStart, tEnd];
            qx.filterESS = pf.ess;
            qx.globalTimes = (tStart:tEnd)';
        end

        function xPath = backwardSimulateAuxPath(obj, model, pf, Uwin, t0, tau)
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
                Xk = squeeze(pf.particles(k,:,:));
                if size(Xk,1) ~= Np, Xk = Xk'; end

                wk = pf.weights(k,:)';
                wk = wk / sum(wk);
                uk = Uwin(k,:);
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

        function pf = runAuxiliaryBootstrapFilter(obj, model, Ywin, Uwin, globalStartIdx)
            T  = size(Ywin,1);
            Dx = obj.numStates;
            Np = obj.numParticles;

            particles = zeros(T, Np, Dx);
            weights   = zeros(T, Np);
            ancestors = zeros(T, Np);
            essHist   = zeros(T,1);

            if globalStartIdx == 1 && ismethod(model, 'sampleInitialStatePrior')
                X0 = model.sampleInitialStatePrior(Np);
            else
                if ismethod(model, 'sampleInitialStatePrior')
                    X0 = model.sampleInitialStatePrior(Np);
                else
                    X0 = obj.initStd * randn(Np, Dx);
                end
                if size(Ywin,2) >= Dx
                    X0 = 0.5*X0 + 0.5*repmat(Ywin(1,1:Dx), Np, 1);
                end
            end

            logw0 = model.obs.particleLogLikelihood(Ywin(1,:), X0);
            w0 = obj.normaliseLogWeights(logw0);
            particles(1,:,:) = X0;
            weights(1,:) = w0;
            ancestors(1,:) = 1:Np;
            essHist(1) = 1 / sum(w0.^2);

            for t = 1:T-1
                wt = weights(t,:)';
                ess = 1 / sum(wt.^2);
                essHist(t) = ess;

                Xprev = squeeze(particles(t,:,:));
                if size(Xprev,1) ~= Np, Xprev = Xprev'; end

                if ess < obj.resampleThreshold * Np
                    aidx = obj.systematicResample(wt);
                    Xres = Xprev(aidx,:);
                    wres = ones(Np,1) / Np;
                else
                    aidx = (1:Np)';
                    Xres = Xprev;
                    wres = wt;
                end

                ut = Uwin(t,:);
                XU = [Xres, repmat(ut, Np, 1)];
                [mAux, extraLogWeight] = model.getAuxiliaryTransitionInfo(XU);
                Xnext = obj.sampleFromTransitionMeanQ(mAux, model.thetaQ);

                logwObs = model.obs.particleLogLikelihood(Ywin(t+1,:), Xnext);
                logw = log(wres + 1e-300) + extraLogWeight + logwObs;
                wnext = obj.normaliseLogWeights(logw);

                particles(t+1,:,:) = Xnext;
                weights(t+1,:) = wnext;
                ancestors(t+1,:) = aidx;
            end
            essHist(T) = 1 / sum(weights(T,:)'.^2);

            pf = struct();
            pf.particles = particles;
            pf.weights   = weights;
            pf.ancestors = ancestors;
            pf.ess       = essHist;
        end

        function Xnext = sampleFromTransitionMeanQ(obj, meanX, Q)
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
                Lq = chol(Q + 1e-9*eye(Dx), 'lower');
                Xnext = meanX + randn(N, Dx) * Lq';
            end
        end
    end

    methods (Static)
        function logp = gaussianDiagLogPdfRows(X, Mu, Var)
            diff2 = (X - Mu).^2;
            logp = -0.5 * sum(log(2*pi*Var) + diff2 ./ Var, 2);
            bad = ~isfinite(logp);
            if any(bad), logp(bad) = -1e12; end
        end

        function logp = gaussianFullLogPdfRows(X, Mu, Sigma)
            [~, D] = size(X);
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
            if isempty(k), k = numel(w); end
        end
    end
end
