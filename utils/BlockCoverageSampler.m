classdef BlockCoverageSampler
    % BlockCoverageSampler
    % Block-wise coverage sampler for
    % 1) plant time-series blocks: (y, u)
    % 2) regressor-sequence blocks: (X, optional Yaux)
    %
    % Main usage:
    %   sampler = BlockCoverageSampler(blockLen, numBlocksKeep, true);
    %   [idxKeep, sampler] = sampler.selectBlocksFromRegressor(Xtr, Ytr);
    %
    % Only test() is static, as requested.

    properties
        % user options
        blockLen
        numBlocksKeep
        useOverlap logical = true

        % internal results
        blockStarts
        blockRanges
        blockFeatures
        blockFeaturesN
        featureMean
        featureStd
        idxBlockSel

        % bookkeeping
        mode char = ''   % 'plant' or 'regressor'
        numCandidates = 0
    end

    methods
        function obj = BlockCoverageSampler(blockLen, numBlocksKeep, useOverlap)
            if nargin < 1 || isempty(blockLen)
                blockLen = 100;
            end
            if nargin < 2 || isempty(numBlocksKeep)
                numBlocksKeep = 5;
            end
            if nargin < 3 || isempty(useOverlap)
                useOverlap = true;
            end

            validateattributes(blockLen, {'numeric'}, ...
                {'scalar','integer','>=',2}, mfilename, 'blockLen');
            validateattributes(numBlocksKeep, {'numeric'}, ...
                {'scalar','integer','>=',1}, mfilename, 'numBlocksKeep');

            obj.blockLen = blockLen;
            obj.numBlocksKeep = numBlocksKeep;
            obj.useOverlap = logical(useOverlap);
        end

        function [idxKeepOriginal, obj] = selectBlocks(obj, y, u)
            % selectBlocks
            % Plant time-series block selection using y and u.

            if ~ismatrix(y) || ~ismatrix(u)
                error('y and u must be 2D matrices.');
            end
            if size(y,1) ~= size(u,1)
                error('y and u must have the same number of rows.');
            end

            N = size(y,1);
            obj = obj.resetInternal();
            obj.mode = 'plant';

            [blockStarts, blockRanges] = obj.buildCandidateBlocks(N);
            numBlocks = numel(blockStarts);

            F = zeros(numBlocks, obj.getPlantFeatureDim(y, u));

            for b = 1:numBlocks
                i1 = blockRanges(b,1);
                i2 = blockRanges(b,2);

                yb = y(i1:i2,:);
                ub = u(i1:i2,:);

                F(b,:) = obj.computeBlockFeature(yb, ub);
            end

            obj.blockStarts = blockStarts(:);
            obj.blockRanges = blockRanges;
            obj.blockFeatures = F;
            [obj.blockFeaturesN, obj.featureMean, obj.featureStd] = obj.normalizeRows(F);

            nKeep = min(obj.numBlocksKeep, numBlocks);
            obj.idxBlockSel = obj.farthestPointSamplingRows(obj.blockFeaturesN, nKeep);
            obj.idxBlockSel = sort(obj.idxBlockSel(:), 'ascend');
            obj.numCandidates = numBlocks;

            idxKeepOriginal = obj.expandSelectedBlocksToIndices();
        end

        function [idxKeep, obj] = selectBlocksFromRegressor(obj, X, Yaux)
            % selectBlocksFromRegressor
            % Regressor-sequence block selection using X and optional Yaux.
            %
            % Inputs
            %   X    : N x D regressor matrix
            %   Yaux : optional N x ny target matrix, can be []

            if nargin < 3
                Yaux = [];
            end

            if ~ismatrix(X)
                error('X must be a 2D matrix.');
            end
            if ~isempty(Yaux) && size(X,1) ~= size(Yaux,1)
                error('X and Yaux must have the same number of rows.');
            end

            N = size(X,1);
            obj = obj.resetInternal();
            obj.mode = 'regressor';

            [blockStarts, blockRanges] = obj.buildCandidateBlocks(N);
            numBlocks = numel(blockStarts);

            F = zeros(numBlocks, obj.getRegressorFeatureDim(X, Yaux));

            for b = 1:numBlocks
                i1 = blockRanges(b,1);
                i2 = blockRanges(b,2);

                Xb = X(i1:i2,:);
                if isempty(Yaux)
                    Yb = [];
                else
                    Yb = Yaux(i1:i2,:);
                end

                F(b,:) = obj.computeRegressorBlockFeature(Xb, Yb);
            end

            obj.blockStarts = blockStarts(:);
            obj.blockRanges = blockRanges;
            obj.blockFeatures = F;
            [obj.blockFeaturesN, obj.featureMean, obj.featureStd] = obj.normalizeRows(F);

            nKeep = min(obj.numBlocksKeep, numBlocks);
            obj.idxBlockSel = obj.farthestPointSamplingRows(obj.blockFeaturesN, nKeep);
            obj.idxBlockSel = sort(obj.idxBlockSel(:), 'ascend');
            obj.numCandidates = numBlocks;

            idxKeep = obj.expandSelectedBlocksToIndices();
        end

        function plotSelectedBlocks(obj, t, y, titleStr)
            % plotSelectedBlocks
            % Visualization for plant-time-series block selection.

            if nargin < 4 || isempty(titleStr)
                titleStr = 'Selected plant blocks';
            end

            if isempty(obj.idxBlockSel)
                error('No selected blocks found. Run selectBlocks first.');
            end
            if isempty(y)
                error('y must not be empty.');
            end
            if numel(t) ~= size(y,1)
                error('Length of t must match number of rows of y.');
            end

            figure;
            plot(t, y(:,1), 'k-', 'LineWidth', 1.0); hold on;
            yl = ylim;

            for k = 1:numel(obj.idxBlockSel)
                b  = obj.idxBlockSel(k);
                i1 = obj.blockRanges(b,1);
                i2 = obj.blockRanges(b,2);

                patch([t(i1) t(i2) t(i2) t(i1)], ...
                      [yl(1) yl(1) yl(2) yl(2)], ...
                      [0.85 0.90 1.00], ...
                      'EdgeColor', 'none', ...
                      'FaceAlpha', 0.25);
            end

            plot(t, y(:,1), 'k-', 'LineWidth', 1.0);
            xlabel('Time');
            ylabel('y_1');
            title(titleStr);
            grid on;
            hold off;
        end

        function plotSelectedRegressorBlocks(obj, X, colIdx, titleStr)
            % plotSelectedRegressorBlocks
            % Visualization for regressor-sequence block selection.

            if nargin < 3 || isempty(colIdx)
                colIdx = 1;
            end
            if nargin < 4 || isempty(titleStr)
                titleStr = 'Selected regressor blocks';
            end

            if isempty(obj.idxBlockSel)
                error('No selected blocks found. Run selectBlocksFromRegressor first.');
            end
            if colIdx < 1 || colIdx > size(X,2)
                error('colIdx is out of range.');
            end

            n = size(X,1);
            xAxis = (1:n).';

            figure;
            plot(xAxis, X(:,colIdx), 'k-', 'LineWidth', 1.0); hold on;
            yl = ylim;

            for k = 1:numel(obj.idxBlockSel)
                b  = obj.idxBlockSel(k);
                i1 = obj.blockRanges(b,1);
                i2 = obj.blockRanges(b,2);

                patch([i1 i2 i2 i1], ...
                      [yl(1) yl(1) yl(2) yl(2)], ...
                      [0.85 0.90 1.00], ...
                      'EdgeColor', 'none', ...
                      'FaceAlpha', 0.25);
            end

            plot(xAxis, X(:,colIdx), 'k-', 'LineWidth', 1.0);
            xlabel('Regressor row index');
            ylabel(sprintf('X(:,%d)', colIdx));
            title(titleStr);
            grid on;
            hold off;
        end

        function summary(obj, Ntotal)
            % summary
            % Print summary to command window.

            if isempty(obj.idxBlockSel)
                fprintf('No selection has been performed yet.\n');
                return;
            end

            idxKeep = obj.expandSelectedBlocksToIndices();

            fprintf('=== BlockCoverageSampler summary ===\n');
            fprintf('Mode                  : %s\n', obj.mode);
            fprintf('Candidate blocks      : %d\n', obj.numCandidates);
            fprintf('Selected blocks       : %d\n', numel(obj.idxBlockSel));
            fprintf('Block length          : %d\n', obj.blockLen);
            fprintf('Overlap enabled       : %d\n', obj.useOverlap);
            fprintf('Selected samples/rows : %d', numel(idxKeep));

            if nargin >= 2 && ~isempty(Ntotal)
                fprintf(' / %d (%.2f%%)\n', Ntotal, 100*numel(idxKeep)/Ntotal);
            else
                fprintf('\n');
            end
        end
    end

    methods (Access = private)
        function obj = resetInternal(obj)
            obj.blockStarts   = [];
            obj.blockRanges   = [];
            obj.blockFeatures = [];
            obj.blockFeaturesN = [];
            obj.featureMean   = [];
            obj.featureStd    = [];
            obj.idxBlockSel   = [];
            obj.mode          = '';
            obj.numCandidates = 0;
        end

        function [blockStarts, blockRanges] = buildCandidateBlocks(obj, N)
            if obj.blockLen > N
                error('blockLen must not exceed sequence length.');
            end

            if obj.useOverlap
                step = max(1, floor(obj.blockLen/2));
            else
                step = obj.blockLen;
            end

            blockStarts = 1:step:(N-obj.blockLen+1);

            if isempty(blockStarts)
                error('No candidate blocks were created.');
            end

            numBlocks = numel(blockStarts);
            blockRanges = zeros(numBlocks, 2);

            for b = 1:numBlocks
                i1 = blockStarts(b);
                i2 = i1 + obj.blockLen - 1;
                blockRanges(b,:) = [i1, i2];
            end
        end

        function d = getPlantFeatureDim(~, y, u)
            ny = size(y,2);
            nu = size(u,2);

            % [meanY stdY minY maxY meanU stdU minU maxU dyRMS duRMS yFirst yLast uFirst uLast]
            d = 4*ny + 4*nu + ny + nu + ny + ny + nu + nu;
        end

        function d = getRegressorFeatureDim(~, X, Yaux)
            Dx = size(X,2);
            if isempty(Yaux)
                Dy = 0;
            else
                Dy = size(Yaux,2);
            end

            % [meanX stdX dXRMS meanY stdY]
            d = Dx + Dx + Dx + Dy + Dy;
        end

        function fb = computeBlockFeature(~, yb, ub)
            % Plant block feature

            featMeanY = mean(yb, 1);
            featStdY  = std(yb, 0, 1);
            featMinY  = min(yb, [], 1);
            featMaxY  = max(yb, [], 1);

            featMeanU = mean(ub, 1);
            featStdU  = std(ub, 0, 1);
            featMinU  = min(ub, [], 1);
            featMaxU  = max(ub, [], 1);

            dy = diff(yb, 1, 1);
            du = diff(ub, 1, 1);

            if isempty(dy)
                featDyRMS = zeros(1, size(yb,2));
            else
                featDyRMS = sqrt(mean(dy.^2, 1));
            end

            if isempty(du)
                featDuRMS = zeros(1, size(ub,2));
            else
                featDuRMS = sqrt(mean(du.^2, 1));
            end

            featYFirst = yb(1,:);
            featYLast  = yb(end,:);
            featUFirst = ub(1,:);
            featULast  = ub(end,:);

            fb = [ ...
                featMeanY, featStdY, featMinY, featMaxY, ...
                featMeanU, featStdU, featMinU, featMaxU, ...
                featDyRMS, featDuRMS, ...
                featYFirst, featYLast, featUFirst, featULast];
        end

        function fb = computeRegressorBlockFeature(~, Xb, Yb)
            % Regressor block feature

            featMeanX = mean(Xb, 1);
            featStdX  = std(Xb, 0, 1);

            dX = diff(Xb, 1, 1);
            if isempty(dX)
                featDXRMS = zeros(1, size(Xb,2));
            else
                featDXRMS = sqrt(mean(dX.^2, 1));
            end

            if isempty(Yb)
                featMeanY = [];
                featStdY  = [];
            else
                featMeanY = mean(Yb, 1);
                featStdY  = std(Yb, 0, 1);
            end

            fb = [featMeanX, featStdX, featDXRMS, featMeanY, featStdY];
        end

        function [Xn, mu, stdv] = normalizeRows(~, X)
            mu = mean(X, 1);
            stdv = std(X, 0, 1);
            stdv(stdv < 1e-12) = 1;
            Xn = (X - mu) ./ stdv;
        end

        function idxSel = farthestPointSamplingRows(~, X, nSel)
            N = size(X,1);

            if nSel > N
                nSel = N;
            end

            idxSel = zeros(nSel,1);

            % Start from point closest to global mean
            xMean = mean(X,1);
            d2Mean = sum((X - xMean).^2, 2);
            [~, idxSel(1)] = min(d2Mean);

            dMin = sum((X - X(idxSel(1),:)).^2, 2);

            for k = 2:nSel
                [~, idxSel(k)] = max(dMin);
                dNew = sum((X - X(idxSel(k),:)).^2, 2);
                dMin = min(dMin, dNew);
            end

            idxSel = unique(idxSel, 'stable');

            % Rare fallback
            if numel(idxSel) < nSel
                remaining = setdiff((1:N)', idxSel, 'stable');
                idxSel = [idxSel; remaining(1:(nSel-numel(idxSel)))];
            end
        end

        function idxKeep = expandSelectedBlocksToIndices(obj)
            idxKeep = [];

            for k = 1:numel(obj.idxBlockSel)
                b = obj.idxBlockSel(k);
                i1 = obj.blockRanges(b,1);
                i2 = obj.blockRanges(b,2);
                idxKeep = [idxKeep; (i1:i2)']; %#ok<AGROW>
            end

            idxKeep = unique(idxKeep, 'stable');
        end
    end

    methods (Static)
        function test()
            rng(1);

            %% Synthetic plant time-series
            Ts = 0.05;
            N  = 1200;
            t  = (0:N-1)' * Ts;

            nu = 2;
            ny = 3;

            u = zeros(N, nu);
            y = zeros(N, ny);

            u(:,1) = 0.8*sin(0.15*t) + 0.3*sin(0.7*t);
            u(:,2) = 0.7 * sign(sin(0.08*t)) + 0.15*randn(N,1);

            y(:,1) = 0.5*sin(0.10*t) + 0.15*u(:,1);
            y(:,2) = 0.4*cos(0.07*t) + 0.20*u(:,2);
            y(:,3) = 0.3*sin(0.12*t + 0.4) + 0.10*u(:,1) - 0.05*u(:,2);

            y(301:600,1) = 1.8*y(301:600,1) + 0.15;
            y(601:900,2) = 1.6*y(601:900,2) - 0.10;
            y(901:end,3) = 2.0*y(901:end,3) + 0.20;

            y = y + 0.03*randn(N,ny);

            %% Plant block test
            sampler1 = BlockCoverageSampler(120, 6, true);
            [idxKeepPlant, sampler1] = sampler1.selectBlocks(y, u);
            sampler1.summary(N);
            sampler1.plotSelectedBlocks(t, y, 'Test: selected plant blocks');

            assert(~isempty(idxKeepPlant), 'Plant block selection failed.');
            assert(all(idxKeepPlant >= 1) && all(idxKeepPlant <= N), ...
                'Plant selected indices out of range.');

            %% Synthetic regressor test
            % Build a simple pseudo-regressor sequence from y,u
            X = [y(3:end,:), y(2:end-1,:), u(2:end-1,:), u(1:end-2,:)];
            Yaux = y(3:end,:);

            sampler2 = BlockCoverageSampler(80, 5, true);
            [idxKeepReg, sampler2] = sampler2.selectBlocksFromRegressor(X, Yaux);
            sampler2.summary(size(X,1));
            sampler2.plotSelectedRegressorBlocks(X, 1, 'Test: selected regressor blocks');

            assert(~isempty(idxKeepReg), 'Regressor block selection failed.');
            assert(all(idxKeepReg >= 1) && all(idxKeepReg <= size(X,1)), ...
                'Regressor selected indices out of range.');

            fprintf('BlockCoverageSampler.test() completed successfully.\n');
        end
    end
end