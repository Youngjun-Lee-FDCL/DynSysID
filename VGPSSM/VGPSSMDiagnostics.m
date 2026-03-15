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

            obj.history.stateMeanMiniBatch = {};
            obj.history.yhatMiniBatch = {};
            obj.history.ttMiniBatch = {};
        end

        function update(obj, model, iter, rho, gamma, qx)
            %---------------------------------
            % q(u) summary
            %---------------------------------
            [muU, SigmaU] = model.qU.getMoments();

            quMeanNorm = norm(muU);
            quCovTrace = trace(SigmaU);

            %---------------------------------
            % Observation fit on minibatch
            %---------------------------------
            tt = qx.tt;
            Ybatch = model.Y(tt, :);

            Xmean = stableMeanTrajectory(qx.Xt);           % [L x Dx]
            Yhat  = model.obs.predict(Xmean);              % [L x Dy]
            obsLL = model.obs.loglik(Ybatch, Xmean);

            %---------------------------------
            % Dynamics residual on minibatch
            %---------------------------------
            Xtp1Mean = stableMeanTrajectory(qx.Xtp1);      % [L x Dx]
            L = numel(tt);

            dynResAccum = 0;
            for i = 1:L
                xt = Xmean(i,:);
                ut = model.U(tt(i),:);

                XU = [xt, ut];
                [mf, ~] = model.predictDynamics(XU);       % [1 x Dx]

                r = Xtp1Mean(i,:) - mf;
                dynResAccum = dynResAccum + norm(r)^2;
            end
            dynResidual = dynResAccum / L;

            %---------------------------------
            % Save history
            %---------------------------------
            obj.history.iter(end+1,1) = iter;
            obj.history.rho(end+1,1) = rho;
            obj.history.gamma(end+1,1) = gamma;

            obj.history.C11(end+1,1) = model.obs.C(1,1);
            obj.history.Cnorm(end+1,1) = norm(model.obs.C, 'fro');

            obj.history.quMeanNorm(end+1,1) = quMeanNorm;
            obj.history.quCovTrace(end+1,1) = quCovTrace;

            obj.history.obsLogLikMiniBatch(end+1,1) = obsLL;
            obj.history.dynResidualMiniBatch(end+1,1) = dynResidual;

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

            fprintf('[Diagnostics] iter=%d | C11=%.4f | ||C||=%.4f | ||mu_u||=%.4f | tr(Sigma_u)=%.4f | obsLL=%.4f | dynRes=%.4f\n', ...
                obj.history.iter(k), ...
                obj.history.C11(k), ...
                obj.history.Cnorm(k), ...
                obj.history.quMeanNorm(k), ...
                obj.history.quCovTrace(k), ...
                obj.history.obsLogLikMiniBatch(k), ...
                obj.history.dynResidualMiniBatch(k));
        end

        function plotTrainingCurves(obj)
            if isempty(obj.history.iter)
                warning('VGPSSMDiagnostics: no history to plot.');
                return;
            end

            figure;

            subplot(2,2,1);
            plot(obj.history.iter, obj.history.obsLogLikMiniBatch, 'LineWidth', 1.5);
            grid on;
            title('Mini-batch Observation Log-Likelihood');
            xlabel('Iteration');

            subplot(2,2,2);
            plot(obj.history.iter, obj.history.dynResidualMiniBatch, 'LineWidth', 1.5);
            grid on;
            title('Mini-batch Dynamics Residual');
            xlabel('Iteration');

            subplot(2,2,3);
            plot(obj.history.iter, obj.history.C11, 'LineWidth', 1.5); hold on;
            plot(obj.history.iter, obj.history.Cnorm, 'LineWidth', 1.5);
            grid on;
            title('Observation Matrix Diagnostics');
            xlabel('Iteration');
            legend('C(1,1)', '||C||_F');

            subplot(2,2,4);
            plot(obj.history.iter, obj.history.quMeanNorm, 'LineWidth', 1.5); hold on;
            plot(obj.history.iter, obj.history.quCovTrace, 'LineWidth', 1.5);
            grid on;
            title('q(u) Diagnostics');
            xlabel('Iteration');
            legend('||mu_u||', 'trace(\Sigma_u)');
        end

        function plotLatestMiniBatchFit(obj, model)
            k = numel(obj.history.iter);
            if k < 1
                warning('VGPSSMDiagnostics: no history to plot.');
                return;
            end

            tt   = obj.history.ttMiniBatch{k};
            Xmean = obj.history.stateMeanMiniBatch{k};
            Yhat = obj.history.yhatMiniBatch{k};
            Ybatch = model.Y(tt,:);

            Dy = size(Ybatch,2);

            figure;
            for j = 1:Dy
                subplot(Dy,1,j);
                plot(tt, Ybatch(:,j), 'k-', 'LineWidth', 1.2); hold on;
                plot(tt, Yhat(:,j), 'r--', 'LineWidth', 1.2);
                grid on;
                xlabel('t');
                ylabel(sprintf('y_%d', j));
                legend('Observed', 'Predicted');
            end
        end

        function plotLatestStateMean(obj)
            k = numel(obj.history.iter);
            if k < 1
                warning('VGPSSMDiagnostics: no history to plot.');
                return;
            end

            tt = obj.history.ttMiniBatch{k};
            Xmean = obj.history.stateMeanMiniBatch{k};
            Dx = size(Xmean,2);

            figure;
            for j = 1:Dx
                subplot(Dx,1,j);
                plot(tt, Xmean(:,j), 'LineWidth', 1.2);
                grid on;
                xlabel('t');
                ylabel(sprintf('x_%d mean', j));
            end
        end
    end
end