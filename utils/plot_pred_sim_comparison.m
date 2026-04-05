function plot_pred_sim_comparison(t, yTrue, yPred, ySim, modelName)

    [N, ny] = size(yTrue);

    figure;
    tiledlayout(ny,1);

    for k = 1:ny
        nexttile;
        plot(t, yTrue(:,k), 'k', 'LineWidth', 1.5); hold on;
        plot(t, yPred(:,k), 'b--', 'LineWidth', 1.2);
        plot(t, ySim(:,k),  'r-.', 'LineWidth', 1.2);

        grid on;
        ylabel(sprintf('y_%d',k));

        if k == 1
            title([modelName, ' | Prediction vs Simulation']);
            legend('True','1-step Pred','Free-run Sim','Location','best');
        end
    end

    xlabel('Time');
end