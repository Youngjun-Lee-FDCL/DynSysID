function plot_pred_sim_per_experiment(tCell, yTrueCell, yPredCell, ySimCell, modelName)

    numExp = numel(tCell);
    ny = size(yTrueCell{1},2);

    for e = 1:numExp
        t = tCell{e};
        yTrue = yTrueCell{e};
        yPred = yPredCell{e};
        ySim  = ySimCell{e};

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
                title(sprintf('%s | Exp %d', modelName, e));
                legend('True','1-step Pred','Free-run Sim','Location','best');
            end
        end

        xlabel('Time');
    end
end