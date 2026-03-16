function print_sysid_summary( ...
    modelName, algoName, nu, ny, ...
    extraInfo)
% Print summary metrics.

fprintf('\n');
fprintf('Model              : %s\n', modelName);
fprintf('Algorithm          : %s\n', algoName);

if ~isempty(extraInfo)
    fprintf('%s\n', extraInfo);
end

fprintf('Number of inputs   : %d\n', nu);
fprintf('Number of outputs  : %d\n', ny);
fprintf('\n');
end