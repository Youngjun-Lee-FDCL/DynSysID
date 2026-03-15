function print_sysid_summary( ...
    modelName, algoName, nu, ny, ...
    rmse1, fit1, rmseFree, fitFree, ...
    rmse1_each, fit1_each, rmseFree_each, fitFree_each, extraInfo)
% Print summary metrics.

fprintf('\n');
fprintf('Model              : %s\n', modelName);
fprintf('Algorithm          : %s\n', algoName);

if ~isempty(extraInfo)
    fprintf('%s\n', extraInfo);
end

fprintf('Number of inputs   : %d\n', nu);
fprintf('Number of outputs  : %d\n', ny);
fprintf('Mean One-step RMSE : %.6f\n', rmse1);
fprintf('Mean One-step FIT  : %.2f %%\n', fit1);
fprintf('Mean Free-run RMSE : %.6f\n', rmseFree);
fprintf('Mean Free-run FIT  : %.2f %%\n', fitFree);

for j = 1:ny
    fprintf('  y_%d -> one-step RMSE %.6f, FIT %.2f %% | free-run RMSE %.6f, FIT %.2f %%\n', ...
        j, rmse1_each(j), fit1_each(j), rmseFree_each(j), fitFree_each(j));
end
fprintf('\n');
end