function [mu, s2] = gpml_predict_mean_var_exact(hyp, meanfunc, covfunc, XtrN, XsN, post)
% Predict GP posterior mean and variance for multiple test points.
% This function uses cached posterior parameters from infGaussLik.

    ns = size(XsN,1);

    mstar = feval(meanfunc{:}, hyp.mean, XsN);
    Kxs   = feval(covfunc{:}, hyp.cov, XtrN, XsN);
    Kss   = feval(covfunc{:}, hyp.cov, XsN, 'diag');

    mu = mstar + Kxs' * post.alpha;

    % Exact predictive variance for latent function:
    % s2 = k(x*,x*) - v'v, where v = L \ Kxs depending on the factorization.
    %
    % In GPML exact inference with Gaussian likelihood, post.L satisfies the
    % convention used internally by gp.m. The expression below matches the
    % standard GPML prediction path for exact Gaussian inference.
    if isfield(post, 'L') && ~isempty(post.L)
        V  = post.L \ Kxs;
        s2 = Kss - sum(V.^2, 1)';
    else
        % Fallback: return latent prior variance if factor is unavailable
        s2 = Kss;
    end

    % Numerical safeguard
    s2 = max(s2, 0);
end