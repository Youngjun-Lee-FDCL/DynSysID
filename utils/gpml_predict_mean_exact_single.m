function mu = gpml_predict_mean_exact_single(hyp, meanfunc, covfunc, XtrN, xstarN, post)
% Predict GP posterior mean at a single test point using cached posterior.

    mstar = feval(meanfunc{:}, hyp.mean, xstarN);
    kstar = feval(covfunc{:}, hyp.cov, XtrN, xstarN);

    mu = mstar + kstar' * post.alpha;
end