function trainSet = build_gp_narx_training_set(data, naMat, nbMat, nkMat, opt)

    Xtr = [];
    Ytr = [];

    numRowsBeforeSampling = 0;
    numRowsAfterSampling  = 0;

    idxKeepPerExp = cell(data.numExp,1);
    idxAllPerExp  = cell(data.numExp,1);

    for e = 1:data.numExp
        [uEstRaw, yEstRaw] = force_2d_io(data.uEstCell{e}, data.yEstCell{e});
        ny = size(yEstRaw,2);
        Ne = size(yEstRaw,1);

        ue = uEstRaw;
        ye = yEstRaw;

        if opt.usePreprocessing
            frameNow = min(opt.sgolayFrame_y, Ne);
            if mod(frameNow,2) == 0
                frameNow = frameNow - 1;
            end
            if frameNow < 3
                error('Savitzky-Golay frame length for output is too short.');
            end

            for j = 1:ny
                ye(:,j) = smoothdata(yEstRaw(:,j), 'sgolay', frameNow);
            end
        end

        [Xe, Ye, idxAll_e] = build_mimo_narx_regressors(ye, ue, naMat, nbMat, nkMat);

        idxAllPerExp{e} = idxAll_e;
        numRowsBeforeSampling = numRowsBeforeSampling + size(Xe,1);

        if opt.useBlockCoverageSampling
            sampler = BlockCoverageSampler( ...
                opt.blockLenSampling, ...
                opt.numBlocksKeepSampling, ...
                opt.useOverlapSampling);

            [idxKeepTrain, ~] = sampler.selectBlocksFromRegressor(Xe, Ye);

            idxKeepPerExp{e} = idxKeepTrain;
            Xe = Xe(idxKeepTrain,:);
            Ye = Ye(idxKeepTrain,:);
        else
            idxKeepPerExp{e} = (1:size(Xe,1)).';
        end

        numRowsAfterSampling = numRowsAfterSampling + size(Xe,1);

        Xtr = [Xtr; Xe];
        Ytr = [Ytr; Ye];
    end

    trainSet = struct();
    trainSet.Xtr = Xtr;
    trainSet.Ytr = Ytr;
    trainSet.idxKeepPerExp = idxKeepPerExp;
    trainSet.idxAllPerExp  = idxAllPerExp;
    trainSet.numRowsBeforeSampling = numRowsBeforeSampling;
    trainSet.numRowsAfterSampling  = numRowsAfterSampling;
end