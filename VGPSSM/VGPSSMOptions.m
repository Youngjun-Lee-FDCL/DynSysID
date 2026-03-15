classdef VGPSSMOptions
    properties
        % Model dimensions
        numStates (1,1) double = 2
        numInducingPoints (1,1) double = 20

        % Inference
        numParticles (1,1) double = 200
        fixedLag (1,1) double = 5
        localSmoothingBuffer (1,1) double = 10
        useLocalWindow logical = true

        % Training
        maxIter (1,1) double = 100
        miniBatchLength (1,1) double = 200

        % Update schedule
        rhoInit (1,1) double = 0.8
        rhoFinalScale (1,1) double = 0.4
        gammaObs (1,1) double = 0.006
        obsUpdateStartIter (1,1) double = 20

        gammaQ (1,1) double = 0.10
        gammaR (1,1) double = 0.10
        gammaHyp (1,1) double = 5e-3
        thetaUpdateStartIter (1,1) double = 10

        gammaZ (1,1) double = 0.15
        zUpdateStartIter (1,1) double = 5

        % Numerical floors / init
        initStateVar (1,1) double = 0.25
        minObsVar (1,1) double = 1e-4
        minProcessVar (1,1) double = 1e-5
        hypFiniteDiffEps (1,1) double = 1e-3
        jitter (1,1) double = 1e-6

        % GPML settings
        covfunc = @covSEard
        meanfunc = []
        gpHypCov = []

        % Observation model
        obsType char = 'Lin+GaussianDiag'

        % Initialisation
        initStrategy char = 'Default'
        inducingStateInit char = 'data'

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
