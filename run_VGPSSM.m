clear; clc; close all;
rng(1);

%% =========================================================
% 0. GPML path check
%% =========================================================
addpath(genpath('.'));

if exist('covSEard.m','file') ~= 2
    error(['GPML path is not set correctly. ', ...
           'Please add GPML to MATLAB path first.']);
end

%% =========================================================
% 1. Generate synthetic data from example function
%% =========================================================
data = generate_vgpssm_easy_example();

t     = data.t;     % [T x 1]
U     = data.u;     % [T x Du]
Xtrue = data.x;     % [T x Dx]
Y     = data.y;     % [T x Dy]

T  = size(Y,1);
Dx = size(Xtrue,2);
Du = size(U,2);
Dy = size(Y,2);

%% =========================================================
% 2. Build options
%% =========================================================
opts = VGPSSMOptions( ...
    'numStates', Dx, ...
    'numInducingPoints', 15, ...
    'numParticles', 30, ...
    'maxIter', 20, ...
    'miniBatchLength', 50, ...
    'fixedLag', 1, ...
    'verbose', true);

opts.covfunc  = @covSEard;
opts.meanfunc = GPMLDynamics.makeStateIdentityMean(Dx);
opts.gpHypCov = [zeros(Dx+Du,1); log(1.0)];

%% =========================================================
% 3. Train model
%% =========================================================
model = VGPSSM(opts);
model.fit(Y, U);

disp('Learned observation matrix C:')
disp(model.obs.C)

%% =========================================================
% 4. Diagnostics
%% =========================================================
if ~isempty(model.diagnostics)
    model.diagnostics.plotTrainingCurves();
    model.diagnostics.plotLatestMiniBatchFit(model);
    model.diagnostics.plotLatestStateMean();
end

%% =========================================================
% 5. One-step dynamics prediction example
%% =========================================================
xTest = [0.2, -0.1];
uTest = 0.3;
XUtest = [xTest, uTest];

[mf1, vf1] = model.predictNext(XUtest);

disp('One-step dynamics prediction at test point:')
disp('Predictive mean:')
disp(mf1)
disp('Predictive variance:')
disp(vf1)

%% =========================================================
% 6. Mean rollout / sample rollout
%% =========================================================
t0 = 50;
H  = 30;

resMean = model.rolloutFromTrainingIndex(t0, H, 'mean');
resSamp = model.rolloutFromTrainingIndex(t0, H, 'sample');

disp('Size of mean rollout state trajectory:')
disp(size(resMean.X))   % [H+1 x Dx]

disp('Size of mean rollout observation trajectory:')
disp(size(resMean.Y))   % [H+1 x Dy]

%% =========================================================
% 7. Compare rollout to true observation
%% =========================================================
model.plotRolloutComparison(t0, H, 'mean');

%% =========================================================
% 8. Ensemble rollout + uncertainty band
%% =========================================================
numEnsemble = 100;

resEns = model.rolloutEnsembleFromTrainingIndex(t0, H, numEnsemble);

disp('Size of ensemble state trajectories:')
disp(size(resEns.Xens))   % [H+1 x Dx x Ne]

disp('Size of ensemble observation trajectories:')
disp(size(resEns.Yens))   % [H+1 x Dy x Ne]

disp('Size of ensemble observation mean:')
disp(size(resEns.Ymean))  % [H+1 x Dy]

model.plotRolloutUncertainty(t0, H, numEnsemble);
model.plotStateRolloutUncertainty(t0, H, numEnsemble);

%% =========================================================
% 9. Direct custom rollout from user-defined initial state
%% =========================================================
x0_custom = [0, 0];
Ufuture_custom = U(1:H, :);

resCustomMean = model.simulateObserved(x0_custom, Ufuture_custom, H, 'mean');
resCustomSamp = model.simulateObserved(x0_custom, Ufuture_custom, H, 'sample');

figure;
for j = 1:Dy
    subplot(Dy,1,j);
    plot(0:H, resCustomMean.Y(:,j), 'b--', 'LineWidth', 1.5); hold on;
    plot(0:H, resCustomSamp.Y(:,j), 'r-.', 'LineWidth', 1.2);
    grid on;
    xlabel('rollout step');
    ylabel(sprintf('y_%d', j));
    legend('Mean rollout', 'Sample rollout');
    title(sprintf('Custom rollout observation y_%d', j));
end

%% =========================================================
% 10. Plot true latent state vs observed data
%% =========================================================
figure;
for j = 1:Dx
    subplot(Dx,1,j);
    plot(t, Xtrue(:,j), 'k-', 'LineWidth', 1.2); hold on;
    plot(t, Y(:,j), 'b:', 'LineWidth', 1.0);
    grid on;
    xlabel('time');
    ylabel(sprintf('state/obs %d', j));
    legend('True state', 'Observed');
    title(sprintf('Ground-truth state vs observation channel %d', j));
end

disp('Demo finished successfully.');


%% Remove path
rmpath(genpath('.'));