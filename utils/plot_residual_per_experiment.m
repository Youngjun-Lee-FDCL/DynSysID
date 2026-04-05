function plot_residual_per_experiment(tCell, yTrueCell, yPredCell, ySimCell, plotTitlePrefix)

    numExp = numel(tCell);
    ny = size(yTrueCell{1},2);

    for e = 1:numExp
        t = tCell{e};
        ePred = yTrueCell{e} - yPredCell{e};
        eSim  = yTrueCell{e} - ySimCell{e};

        figure;
        tiledlayout(ny,2);

        for k = 1:ny
            nexttile;
            plot(t, ePred(:,k), 'b', 'LineWidth', 1.0);
            grid on;
            title(sprintf('Pred residual y_%d', k));
            ylabel('Error');

            nexttile;
            plot(t, eSim(:,k), 'r', 'LineWidth', 1.0);
            grid on;
            title(sprintf('Sim residual y_%d', k));
        end

        sgtitle(sprintf('%s | Exp %d Residual', plotTitlePrefix, e));
    end
end