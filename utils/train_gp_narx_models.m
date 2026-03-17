function gpModel = train_gp_narx_models(XtrN, YtrN, opt)

    ny = size(YtrN,2);
    D  = size(XtrN,2);

    % Ensure GPML function specifications are cell arrays
    meanfunc = opt.meanfunc;
    covfunc  = opt.covfunc;
    likfunc  = opt.likfunc;
    inffunc  = opt.inffunc;

    if ~iscell(meanfunc), meanfunc = {meanfunc}; end
    if ~iscell(covfunc),  covfunc  = {covfunc};  end
    if ~iscell(likfunc),  likfunc  = {likfunc};  end

    hypOpt   = cell(ny,1);
    postCell = cell(ny,1);

    for j = 1:ny
        hyp = struct();
        hyp.mean = [];
        hyp.cov  = [zeros(D,1); 0];
        hyp.lik  = log(0.1);

        evalc('hypOpt{j} = minimize(hyp, @gp, opt.numMinimizeIter, inffunc, meanfunc, covfunc, likfunc, XtrN, YtrN(:,j));');

        [postCell{j}, ~, ~] = feval(inffunc, ...
            hypOpt{j}, meanfunc, covfunc, likfunc, XtrN, YtrN(:,j));
    end

    gpModel = struct();
    gpModel.hypOpt   = hypOpt;
    gpModel.postCell = postCell;
    gpModel.meanfunc = meanfunc;
    gpModel.covfunc  = covfunc;
    gpModel.likfunc  = likfunc;
    gpModel.inffunc  = inffunc;
end