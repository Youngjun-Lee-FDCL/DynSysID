classdef VGPSSMDiagnostics < handle
    properties
        history
    end

    methods
        function obj = VGPSSMDiagnostics()
            obj.reset();
        end

        function reset(obj)
            obj.history = struct();
            obj.history.iter = [];
            obj.history.rho = [];
            obj.history.gamma = [];
            obj.history.C11 = [];
            obj.history.Cnorm = [];
            obj.history.quMeanNorm = [];
            obj.history.quCovTrace = [];
            obj.history.obsLogLikMiniBatch = [];
            obj.history.dynResidualMiniBatch = [];
            obj.history.approxELBO = [];
            obj.history.processVarMean = [];
            obj.history.obsVarMean = [];
            obj.history.hypNorm = [];
            obj.history.stateMeanMiniBatch = {};
            obj.history.yhatMiniBatch = {};
            obj.history.ttMiniBatch = {};
        end

        function update(obj, model, iter, rho, gamma, qx)
            [muU, SigmaU] = model.qU.getMoments();
            quMeanNorm = norm(muU);
            quCovTrace = trace(SigmaU);

            tt = qx.tt;
            Ybatch = model.Y(tt, :);
            Xmean = stableMeanTrajectory(qx.Xt);
            Yhat  = model.obs.predict(Xmean);
            obsLL = model.obs.loglik(Ybatch, Xmean);

            Xtp1Mean = stableMeanTrajectory(qx.Xtp1);
            L = numel(tt);
            dynResAccum = 0;
            for i = 1:L
                XU = [Xmean(i,:), model.U(tt(i),:)];
                [mf, ~] = model.predictDynamics(XU);
                r = Xtp1Mean(i,:) - mf;
                dynResAccum = dynResAccum + norm(r)^2;
            end
            dynResidual = dynResAccum / max(L,1);

            approxELBO = model.approximateELBO(qx);

            obj.history.iter(end+1,1) = iter;
            obj.history.rho(end+1,1) = rho;
            obj.history.gamma(end+1,1) = gamma;
            obj.history.C11(end+1,1) = model.obs.C(1,1);
            obj.history.Cnorm(end+1,1) = norm(model.obs.C, 'fro');
            obj.history.quMeanNorm(end+1,1) = quMeanNorm;
            obj.history.quCovTrace(end+1,1) = quCovTrace;
            obj.history.obsLogLikMiniBatch(end+1,1) = obsLL;
            obj.history.dynResidualMiniBatch(end+1,1) = dynResidual;
            obj.history.approxELBO(end+1,1) = approxELBO;
            obj.history.processVarMean(end+1,1) = mean(diag(model.thetaQ));
            obj.history.obsVarMean(end+1,1) = mean(diag(model.obs.R));
            obj.history.hypNorm(end+1,1) = norm(model.gp.hyp.cov);
            obj.history.stateMeanMiniBatch{end+1,1} = Xmean;
            obj.history.yhatMiniBatch{end+1,1} = Yhat;
            obj.history.ttMiniBatch{end+1,1} = tt;
        end

        function printLast(obj)
            k = numel(obj.history.iter);
            if k < 1
                fprintf('[Diagnostics] No history available.\n');
                return;
            end
            fprintf(['[Diagnostics] iter=%d | C11=%.4f | ||C||=%.4f | ||mu_u||=%.4f | ' ...
                'tr(Sigma_u)=%.4f | obsLL=%.4f | dynRes=%.4f | ELBO~=%.4f | ' ...
                'meanQ=%.4e | meanR=%.4e\n'], ...
                obj.history.iter(k), obj.history.C11(k), obj.history.Cnorm(k), ...
                obj.history.quMeanNorm(k), obj.history.quCovTrace(k), ...
                obj.history.obsLogLikMiniBatch(k), obj.history.dynResidualMiniBatch(k), ...
                obj.history.approxELBO(k), obj.history.processVarMean(k), ...
                obj.history.obsVarMean(k));
        end

        function plotLatestStateMean(obj)
            k = numel(obj.history.iter);
            if k < 1
                warning('VGPSSMDiagnostics: no history to plot.');
                return;
            end

            tt    = obj.history.ttMiniBatch{k};
            Xmean = obj.history.stateMeanMiniBatch{k};
            Dx    = size(Xmean, 2);

            figure;
            for j = 1:Dx
                subplot(Dx,1,j);
                plot(tt, Xmean(:,j), 'b-', 'LineWidth', 1.5);
                grid on; xlabel('t'); ylabel(sprintf('x_%d', j));
                title('Latest mini-batch posterior state mean');
            end
        end

        function plotTrainingCurves(obj)
            if isempty(obj.history.iter)
                warning('VGPSSMDiagnostics: no history to plot.');
                return;
            end

            figure;
            subplot(3,2,1);
            plot(obj.history.iter, obj.history.obsLogLikMiniBatch, 'LineWidth', 1.5);
            grid on; title('Mini-batch Observation Log-Likelihood'); xlabel('Iteration');

            subplot(3,2,2);
            plot(obj.history.iter, obj.history.dynResidualMiniBatch, 'LineWidth', 1.5);
            grid on; title('Mini-batch Dynamics Residual'); xlabel('Iteration');

            subplot(3,2,3);
            plot(obj.history.iter, obj.history.C11, 'LineWidth', 1.5); hold on;
            plot(obj.history.iter, obj.history.Cnorm, 'LineWidth', 1.5);
            grid on; title('Observation Matrix Diagnostics'); xlabel('Iteration');
            legend('C(1,1)', '||C||_F');

            subplot(3,2,4);
            plot(obj.history.iter, obj.history.quMeanNorm, 'LineWidth', 1.5); hold on;
            plot(obj.history.iter, obj.history.quCovTrace, 'LineWidth', 1.5);
            grid on; title('q(u) Diagnostics'); xlabel('Iteration');
            legend('||mu_u||', 'trace(\Sigma_u)');

            subplot(3,2,5);
            plot(obj.history.iter, obj.history.approxELBO, 'LineWidth', 1.5);
            grid on; title('Approximate ELBO'); xlabel('Iteration');

            subplot(3,2,6);
            plot(obj.history.iter, obj.history.processVarMean, 'LineWidth', 1.5); hold on;
            plot(obj.history.iter, obj.history.obsVarMean, 'LineWidth', 1.5); hold on;
            plot(obj.history.iter, obj.history.hypNorm, 'LineWidth', 1.5);
            grid on; title('Theta Diagnostics'); xlabel('Iteration');
            legend('mean diag(Q)', 'mean diag(R)', '||hyp.cov||');
        end

        function plotLatestMiniBatchFit(obj, model)
            k = numel(obj.history.iter);
            if k < 1
                warning('VGPSSMDiagnostics: no history to plot.');
                return;
            end

            tt   = obj.history.ttMiniBatch{k};
            Yhat = obj.history.yhatMiniBatch{k};
            Ybatch = model.Y(tt,:);
            Dy = size(Ybatch,2);

            figure;
            for j = 1:Dy
                subplot(Dy,1,j);
                plot(tt, Ybatch(:,j), 'k-', 'LineWidth', 1.2); hold on;
                plot(tt, Yhat(:,j), 'r--', 'LineWidth', 1.2);
                grid on; xlabel('t'); ylabel(sprintf('y_%d', j));
                legend('Observed', 'Predicted');
            end
        end
    end
end
