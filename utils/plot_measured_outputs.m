function plot_measured_outputs(t, y, modelName)
% Plot full measured outputs.

    ny = size(y,2);

    figure('Color','w');
    for j = 1:ny
        subplot(ny,1,j);
        plot(t, y(:,j), 'LineWidth', 1.2);
        grid on;
        xlabel('Time (s)');
        ylabel(sprintf('y_%d', j));
        if j == 1
            title(['Measured output - ', modelName]);
        end
    end
end