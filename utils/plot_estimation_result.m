function plot_estimation_result( ...
    tVal, yTrue, yHat, rmse_each, fit_each, ...
    algoName, lineLabel, lineSpec)
% Plot one-step or free-run result.
%
% Supports:
%   - single experiment (matrix)
%   - multi-experiment (cell)

    %% =========================
    % Convert cell -> matrix with NaN gaps
    %% =========================
    if iscell(yTrue)
        [tVal, yTrue] = concat_with_gaps(tVal, yTrue);
        [~,    yHat ] = concat_with_gaps(tVal, yHat);
    end

    ny = size(yTrue, 2);

    %% =========================
    % Plot
    %% =========================
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

    sgtitle(algoName);
end

%% ========================================================================
function [tOut, yOut] = concat_with_gaps(tCell, yCell)
% Concatenate multiple experiments with NaN gaps

    numExp = numel(yCell);
    ny = size(yCell{1},2);

    tOut = [];
    yOut = [];

    for e = 1:numExp
        t_e = tCell{e};
        y_e = yCell{e};

        if isvector(t_e), t_e = t_e(:); end
        if isvector(y_e), y_e = y_e(:); end

        tOut = [tOut; t_e; NaN];
        yOut = [yOut; y_e; NaN(1, ny)];
    end
end