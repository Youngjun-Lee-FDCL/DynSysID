function plot_gp_prediction_with_band( ...
    tVal, yTrue, yPred, yVar, rmse_each, fit_each, ...
    algoName)
% Plot GP prediction with 95% uncertainty band.
%
% Inputs:
%   tVal      : validation time vector, N x 1
%   yTrue     : true outputs, N x ny
%   yPred     : predictive mean, N x ny
%   yVar      : predictive variance, N x ny
%   rmse_each : RMSE for each output channel
%   fit_each  : FIT for each output channel
%   modelName : dataset/model name
%   algoName  : algorithm name, e.g. 'GP-NARX'
%   figLabel  : figure title label

    ny = size(yTrue, 2);

    figure('Color','w');
    for j = 1:ny
        subplot(ny,1,j);

        upper = yPred(:,j) + 2*sqrt(max(yVar(:,j), 0));
        lower = yPred(:,j) - 2*sqrt(max(yVar(:,j), 0));

        fill([tVal; flipud(tVal)], [upper; flipud(lower)], ...
            [0.85 0.90 1.00], 'EdgeColor', 'none'); hold on;
        plot(tVal, yTrue(:,j), 'k', 'LineWidth', 1.2);
        plot(tVal, yPred(:,j), 'r--', 'LineWidth', 1.4);

        grid on;
        xlabel('Time (s)');
        ylabel(sprintf('y_%d', j));
        legend('95% band', 'True', ['GP-NARX', ' one-step'], 'Location', 'best');
        title(sprintf('One-step | y_%d | RMSE = %.4f | FIT = %.2f%%', ...
            j, rmse_each(j), fit_each(j)));
    end

    sgtitle(algoName);
end