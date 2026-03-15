classdef GPMLDynamics < handle
    properties
        % GPML function handles
        covfunc
        meanfunc

        % GPML-style hyperparameter struct
        % hyp.cov : covariance hyperparameters
        % hyp.mean: mean hyperparameters (optional)
        hyp struct

        % Inducing inputs
        Z

        % Model dimensions
        numStates (1,1) double
        inputDim  (1,1) double

        % Numerical stabilisation
        jitter (1,1) double = 1e-6
    end

    methods
        function obj = GPMLDynamics(covfunc, meanfunc, hyp, Z, numStates)
            if nargin < 5
                error('GPMLDynamics: require covfunc, meanfunc, hyp, Z, numStates.');
            end

            obj.covfunc   = covfunc;
            obj.meanfunc  = meanfunc;
            obj.hyp       = hyp;
            obj.Z         = Z;
            obj.numStates = numStates;
            obj.inputDim  = size(Z,2);

            obj.validateSetup();
        end

        function validateSetup(obj)
            if isempty(obj.covfunc)
                error('GPMLDynamics: covfunc is empty.');
            end
            if isempty(obj.meanfunc)
                error('GPMLDynamics: meanfunc is empty.');
            end
            if ~isfield(obj.hyp, 'cov')
                error('GPMLDynamics: hyp.cov is missing.');
            end
            if ~isfield(obj.hyp, 'mean')
                obj.hyp.mean = [];
            end
            if isempty(obj.Z)
                error('GPMLDynamics: inducing inputs Z are empty.');
            end
        end

        function setInducingInputs(obj, Znew)
            obj.Z = Znew;
            obj.inputDim = size(Znew,2);
        end

        function setHypCov(obj, hypCov)
            obj.hyp.cov = hypCov;
        end

        function setHypMean(obj, hypMean)
            obj.hyp.mean = hypMean;
        end

        function K = Kzz(obj)
            K = feval(obj.covfunc, obj.hyp.cov, obj.Z);
            K = obj.addJitter(K);
        end

        function K = Kxz(obj, XU)
            K = feval(obj.covfunc, obj.hyp.cov, XU, obj.Z);
        end

        function Kdiag = KxxDiag(obj, XU)
            Kdiag = feval(obj.covfunc, obj.hyp.cov, XU, 'diag');
            Kdiag = Kdiag(:);
        end

        function dK = dKzz(obj, iHyp)
            % Derivative of Kzz wrt iHyp-th covariance hyperparameter
            dK = feval(obj.covfunc, obj.hyp.cov, obj.Z, [], iHyp);
        end

        function L = cholKzz(obj)
            K = obj.Kzz();
            L = chol(K, 'lower');
        end

        function X = solveKzz(obj, B)
            % Solve Kzz * X = B stably
            L = obj.cholKzz();
            X = L' \ (L \ B);
        end

        function X = solveKzzT(obj, B)
            % Solve X * Kzz = B  =>  X = B / Kzz
            L = obj.cholKzz();
            X = (B / L') / L;
        end

        function A = computeA(obj, XU)
            % A = Kxz / Kzz = Kxz * inv(Kzz)
            Kxz = obj.Kxz(XU);
            A = obj.solveKzzT(Kxz);
        end

        function [Kzz, Kxz, m] = getCoreTerms(obj, XU)
            Kzz = obj.Kzz();
            Kxz = obj.Kxz(XU);
            m   = obj.mean(XU);
        end

        function [mf, vf] = predictFromQu(obj, XU, qU)
            [muMat, Sigma3] = qU.getBlockMoments();

            N  = size(XU,1);
            Dx = obj.numStates;
            M  = size(obj.Z,1);

            if size(muMat,1) ~= M || size(muMat,2) ~= Dx
                error('GPMLDynamics: qU mean block size mismatch.');
            end

            A       = obj.computeA(XU);     % [N x M]
            Kxz     = obj.Kxz(XU);          % [N x M]
            mX      = obj.mean(XU);         % [N x Dx]
            mZ      = obj.mean(obj.Z);      % [M x Dx]
            KxxDiag = obj.KxxDiag(XU);      % [N x 1]

            mf = zeros(N, Dx);
            vf = zeros(N, Dx);

            priorCondVar = max(KxxDiag - sum(A .* Kxz, 2), 1e-12);

            for d = 1:Dx
                mu_d    = muMat(:,d);
                Sigma_d = Sigma3(:,:,d);

                mf(:,d) = mX(:,d) + A * (mu_d - mZ(:,d));
                corrVar = sum((A * Sigma_d) .* A, 2);
                vf(:,d) = max(priorCondVar + corrVar, 1e-12);
            end
        end

        function B = computeB(obj, XU)
            % B(x) = k(x,x) - K_xz K_zz^{-1} K_zx
            %
            % Output:
            %   B : [N x Dx]
            %
            % Here we assume independent GP outputs sharing the same kernel,
            % so the scalar conditional variance is replicated across Dx outputs.

            N  = size(XU,1);
            Dx = obj.numStates;

            Kxz = obj.Kxz(XU);              % [N x M]
            A   = obj.computeA(XU);         % [N x M] = Kxz * Kzz^{-1}
            KxxDiag = obj.KxxDiag(XU);      % [N x 1]

            % diag(Kxz * Kzz^{-1} * Kzx) = row-wise sum(A .* Kxz, 2)
            condVar = KxxDiag - sum(A .* Kxz, 2);
            condVar = max(condVar, 1e-12);

            B = repmat(condVar, 1, Dx);
        end

        function m = mean(obj, XU)
            m = feval(obj.meanfunc, obj.hyp.mean, XU);

            if size(m,1) ~= size(XU,1) || size(m,2) ~= obj.numStates
                error('GPMLDynamics: mean function must return [N x numStates].');
            end
        end

        function S = summary(obj)
            S = struct();
            S.numStates = obj.numStates;
            S.inputDim  = obj.inputDim;
            S.numInducing = size(obj.Z,1);

            if isa(obj.covfunc, 'function_handle')
                S.covfunc = func2str(obj.covfunc);
            else
                S.covfunc = 'non-handle/cell covfunc';
            end

            if isa(obj.meanfunc, 'function_handle')
                S.meanfunc = func2str(obj.meanfunc);
            else
                S.meanfunc = 'non-handle/cell meanfunc';
            end
        end
    end

    methods (Access = private)
        function K = addJitter(obj, K)
            K = K + obj.jitter * eye(size(K));
        end
    end

    methods (Static)
        function hyp = makeDefaultHyp(inputDim, meanHyp)
            % For covSEard:
            % hyp.cov = [log(ell_1); ...; log(ell_D); log(sf)]
            if nargin < 2
                meanHyp = [];
            end
            hyp = struct();
            hyp.cov  = [zeros(inputDim,1); log(1.0)];
            hyp.mean = meanHyp;
        end

        function meanfunc = makeStateIdentityMean(numStates)
            % GPML mean function signature:
            % meanfunc(hypMean, XU) -> [N x Dx]
            meanfunc = @(hypMean, XU) XU(:,1:numStates);
        end
    end

    methods (Static)
        function test()
            Dx = 2;
            Du = 1;
            M  = 10;
            N  = 50;

            covfunc  = @covSEard;
            meanfunc = GPMLDynamics.makeStateIdentityMean(Dx);

            hyp = GPMLDynamics.makeDefaultHyp(Dx + Du);

            Z  = randn(M, Dx+Du);
            XU = randn(N, Dx+Du);

            gp = GPMLDynamics(covfunc, meanfunc, hyp, Z, Dx);

            Kzz = gp.Kzz();
            Kxz = gp.Kxz(XU);
            A   = gp.computeA(XU);
            m   = gp.mean(XU);

            disp(size(Kzz))   % [M M]
            disp(size(Kxz))   % [N M]
            disp(size(A))     % [N M]
            disp(size(m))     % [N Dx]
        end
    end
end