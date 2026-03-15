function [tt, numMiniBatches] = sampleMiniBatch(T, miniBatchLength, edgeLength)
%SAMPLEMINIBATCH Sample a contiguous time minibatch
%
% Inputs
%   T               : total trajectory length
%   miniBatchLength : number of time indices in minibatch
%   edgeLength      : safety margin near boundaries
%
% Outputs
%   tt              : [L x 1] sampled time indices
%   numMiniBatches  : scaling factor ~ T / L

    if T < 2
        error('sampleMiniBatch: T must be at least 2.');
    end

    L = min(miniBatchLength, T-1);

    startMin = 1 + edgeLength;
    startMax = T - L - edgeLength;

    if startMax < startMin
        startMin = 1;
        startMax = T - L;
    end

    if startMax < startMin
        startIdx = 1;
    else
        startIdx = randi([startMin, startMax], 1, 1);
    end

    tt = (startIdx:startIdx+L-1)';
    numMiniBatches = max((T-1) / numel(tt), 1);
end