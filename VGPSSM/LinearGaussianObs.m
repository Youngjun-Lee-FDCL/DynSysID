classdef LinearGaussianObs < handle
    properties
        C
        R
        assumeDiagonalR logical = true
    end

    methods
        function obj = LinearGaussianObs(C, R)
            obj.C = C;
            obj.R = R;
        end

        function Yhat = predict(obj, X)
            Yhat = X * obj.C';
        end

        function res = residual(obj, Y, X)
            Yhat = obj.predict(X);
            res = Y - Yhat;
        end

        function ll = loglik(obj, Y, X)
            res = obj.residual(Y, X);
            Dy = size(Y,2);

            if obj.assumeDiagonalR
                invRdiag = 1 ./ diag(obj.R);
                logdetR = sum(log(diag(obj.R)));
                quad = sum((res.^2) .* invRdiag', 2);
                ll = sum(-0.5 * (Dy*log(2*pi) + logdetR + quad));
            else
                [L,p] = chol(obj.R, 'lower');
                if p ~= 0
                    error('LinearGaussianObs: R must be SPD.');
                end
                XC = res / L';
                quad = sum(XC.^2, 2);
                logdetR = 2 * sum(log(diag(L)));
                ll = sum(-0.5 * (Dy*log(2*pi) + logdetR + quad));
            end
        end

        function logw = particleLogLikelihood(obj, y, Xparticles)
            if isrow(y)
                y = y(:);
            end

            Yhat = obj.predict(Xparticles);
            Res  = Yhat - y';

            Dy = numel(y);

            if obj.assumeDiagonalR
                invRdiag = 1 ./ diag(obj.R);
                logdetR  = sum(log(diag(obj.R)));
                quad = sum((Res.^2) .* invRdiag', 2);
                logw = -0.5 * (Dy*log(2*pi) + logdetR + quad);
            else
                [L,p] = chol(obj.R, 'lower');
                if p ~= 0
                    error('LinearGaussianObs: R must be SPD.');
                end
                XC = Res / L';
                quad = sum(XC.^2, 2);
                logdetR = 2 * sum(log(diag(L)));
                logw = -0.5 * (Dy*log(2*pi) + logdetR + quad);
            end
        end

        function gradC = gradCFromParticles(obj, Y, XtParticles, W)
            [L, Dx, Np] = size(XtParticles);
            Dy = size(Y,2);

            if nargin < 4 || isempty(W)
                W = ones(L, Np) / Np;
            end

            if obj.assumeDiagonalR
                invRdiag = 1 ./ diag(obj.R);
            else
                error('gradCFromParticles currently implemented only for diagonal R.');
            end

            gradC = zeros(Dy, Dx);

            for t = 1:L
                yt = Y(t,:)';

                for p = 1:Np
                    xt = squeeze(XtParticles(t,:,p))';
                    err = yt - obj.C * xt;
                    gradC = gradC + W(t,p) * (invRdiag .* err) * xt';
                end
            end
        end

        function updateC(obj, gradC, gamma)
            obj.C = obj.C + gamma * gradC;
        end
    end
end