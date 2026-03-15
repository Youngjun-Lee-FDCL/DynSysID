function plot_measured_outputs_with_preprocessed(t, yRaw, yFilt, modelName)

    ny = size(yRaw,2);

    figure('Name', [modelName ' - Measured outputs with preprocessing'], 'Color', 'w');
    tiledlayout(ny,1,'TileSpacing','compact','Padding','compact');

    for i = 1:ny
        nexttile;
        plot(t, yRaw(:,i), '-', 'Color', 'r', 'LineWidth', 0.8); hold on;
        plot(t, yFilt(:,i), 'k-', 'LineWidth', 1.2);
        grid on;
        ylabel(sprintf('y_%d', i), 'Interpreter', 'none');
        if i == 1
            title([modelName ' measured outputs (raw vs preprocessed)'], ...
                'Interpreter', 'none');
            legend('Raw','Preprocessed','Location','best');
        end
    end
    xlabel('Time');
end