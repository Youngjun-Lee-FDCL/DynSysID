classdef VariationalGaussian < handle
    properties
        eta1
        eta2
        eta1Old
        eta2Old
        numBlocks (1,1) double = 1
        blockSize (1,1) double = 1
        Sigma
        mu
        jitter (1,1) double = 1e-9
    end

    methods
        function obj = VariationalGaussian(numBlocks, blockSize)
            if nargin >= 1
                obj.numBlocks = numBlocks;
            end
            if nargin >= 2
                obj.blockSize = blockSize;
            end
        end

        function setFromMoments(obj, mu, Sigma)
            n = obj.numBlocks * obj.blockSize;

            if size(Sigma,1) ~= n || size(Sigma,2) ~= n
                error('VariationalGaussian:setFromMoments size mismatch in Sigma.');
            end
            if numel(mu) ~= n
                error('VariationalGaussian:setFromMoments size mismatch in mu.');
            end

            mu = mu(:);
            Sigma = 0.5 * (Sigma + Sigma');
            Sigma = Sigma + obj.jitter * eye(n);

            % Prefer Cholesky over inv for stability
            [L,p] = chol(Sigma, 'lower');
            if p ~= 0
                error('VariationalGaussian:setFromMoments Sigma is not SPD.');
            end

            Precision = L' \ (L \ eye(n));

            obj.mu = mu;
            obj.Sigma = Sigma;
            obj.eta1 = Precision * mu;
            obj.eta2 = -0.5 * Precision;
        end

        function [mu, Sigma] = getMoments(obj)
            % Return cached moments for consistency/speed
            mu = obj.mu;
            Sigma = obj.Sigma;
        end

        function [muMat, Sigma3] = getBlockMoments(obj)
            [mu_, Sigma_] = obj.getMoments();

            M = obj.blockSize;
            D = obj.numBlocks;

            muMat = reshape(mu_, M, D);
            Sigma3 = zeros(M, M, D);

            for d = 1:D
                ii = (d-1)*M + (1:M);
                Sigma3(:,:,d) = Sigma_(ii,ii);
            end
        end

        function saveOld(obj)
            obj.eta1Old = obj.eta1;
            obj.eta2Old = obj.eta2;
        end

        function dampedUpdate(obj, eta1New, eta2New, rho)
            if isempty(obj.eta1) || isempty(obj.eta2)
                error('VariationalGaussian:dampedUpdate natural parameters are not initialised.');
            end

            if rho < 0 || rho > 1
                error('VariationalGaussian:dampedUpdate rho must be in [0,1].');
            end

            obj.saveOld();

            obj.eta1 = (1 - rho) * obj.eta1Old + rho * eta1New;
            obj.eta2 = (1 - rho) * obj.eta2Old + rho * eta2New;

            Precision = -2 * obj.eta2;
            Precision = 0.5 * (Precision + Precision');
            Precision = Precision + obj.jitter * eye(size(Precision));

            [L,p] = chol(Precision, 'lower');
            if p ~= 0
                error('VariationalGaussian:dampedUpdate precision is not SPD.');
            end

            obj.Sigma = L' \ (L \ eye(size(Precision)));
            obj.mu = obj.Sigma * obj.eta1;
        end
    end
end