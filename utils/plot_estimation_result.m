function plot_estimation_result( ...
    tVal, yTrue, yHat, rmse_each, fit_each, ...
    modelName, algoName, figLabel, lineLabel, lineSpec)
% Plot one-step or free-run result.

    ny = size(yTrue, 2);

    figure('Color','w');
    for j = 1:ny
        subplot(ny,1,j);
        plot(tVal, yTrue(:,j), 'k', 'LineWidth', 1.2); hold on;
        plot(tVal, yHat(:,j), lineSpec, 'LineWidth', 1.4);
        grid on;
        xlabel('Time (s)');
        ylabel(sprintf('y_%d', j));
        legend('True', [algoName, ' ', lower(lineLabel)], 'Location', 'best');
        title(sprintf('%s | y_%d | RMSE = %.4f | FIT = %.2f%%', ...
            lineLabel, j, rmse_each(j), fit_each(j)));
    end
    sgtitle([modelName, ' | ', algoName, ' | ', figLabel]);
end