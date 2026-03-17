function plot_estimation_errors(tVal, yTrue, yPred, ySim, algoName)
% Plot one-step and free-run errors.

    ny = size(yTrue, 2);

    figure('Color','w');
    for j = 1:ny
        subplot(ny,2,2*j-1);
        plot(tVal, yTrue(:,j) - yPred(:,j), 'LineWidth', 1.2);
        grid on;
        xlabel('Time (s)');
        ylabel(sprintf('e_{1step,%d}', j));
        title(sprintf('One-step error | y_%d', j));

        subplot(ny,2,2*j);
        plot(tVal, yTrue(:,j) - ySim(:,j), 'LineWidth', 1.2);
        grid on;
        xlabel('Time (s)');
        ylabel(sprintf('e_{free,%d}', j));
        title(sprintf('Free-run error | y_%d', j));
    end
    sgtitle(algoName);
end