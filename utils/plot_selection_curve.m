function plot_selection_curve(candidates, fitPred_mean, fitSim_mean, rmsePred_mean, rmseSim_mean, xLabel, titleStr)
% Plot model-selection curve for candidate settings.
%
% Inputs:
%   candidates    : x-axis values (e.g., model orders)
%   fitPred_mean  : mean one-step FIT for each candidate
%   fitSim_mean   : mean free-run FIT for each candidate
%   rmsePred_mean : mean one-step RMSE for each candidate
%   rmseSim_mean  : mean free-run RMSE for each candidate
%   xLabel        : x-axis label string
%   titleStr      : figure title string

    figure('Color','w');

    yyaxis left;
    plot(candidates, fitPred_mean, 'o-', 'LineWidth', 1.2, 'MarkerSize', 6); hold on;
    plot(candidates, fitSim_mean,  's-', 'LineWidth', 1.2, 'MarkerSize', 6);
    ylabel('Mean FIT (%)');

    yyaxis right;
    plot(candidates, rmsePred_mean, 'o--', 'LineWidth', 1.0, 'MarkerSize', 6);
    plot(candidates, rmseSim_mean,  's--', 'LineWidth', 1.0, 'MarkerSize', 6);
    ylabel('Mean RMSE');

    grid on;
    xlabel(xLabel);
    title(titleStr);
    legend('Mean one-step FIT', 'Mean free-run FIT', ...
           'Mean one-step RMSE', 'Mean free-run RMSE', ...
           'Location', 'best');
end