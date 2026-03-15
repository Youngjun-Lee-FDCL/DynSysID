classdef VGPSSMOptions
    properties
        % Model dimensions
        numStates (1,1) double = 2
        numInducingPoints (1,1) double = 20

        % Inference
        numParticles (1,1) double = 200
        fixedLag (1,1) double = 5

        % Training
        maxIter (1,1) double = 100
        miniBatchLength (1,1) double = 200
        localSmoothingBuffer (1,1) double = 10

        % Update schedule
        rhoInit (1,1) double = 0.8
        rhoFinalScale (1,1) double = 0.4
        gammaObs (1,1) double = 0.006
        obsUpdateStartIter (1,1) double = 50

        % GPML settings
        covfunc = @covSEard
        meanfunc = []
        gpHypCov = []

        % Observation model
        obsType char = 'Lin+GaussianDiag'

        % Initialisation
        initStrategy char = 'Default'

        % Convergence
        convergenceType char = 'FixedIter'

        % Debug
        verbose logical = true
    end

    methods
        function obj = VGPSSMOptions(varargin)
            if mod(nargin,2) ~= 0
                error('VGPSSMOptions: constructor requires name-value pairs.');
            end
            for k = 1:2:nargin
                name = varargin{k};
                value = varargin{k+1};
                if isprop(obj, name)
                    obj.(name) = value;
                else
                    error('VGPSSMOptions: unknown property "%s".', name);
                end
            end

            if isempty(obj.meanfunc)
                obj.meanfunc = GPMLDynamics.makeStateIdentityMean(obj.numStates);
            end
        end
    end
end