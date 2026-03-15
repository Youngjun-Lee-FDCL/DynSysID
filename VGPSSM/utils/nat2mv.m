function [mu, Sigma] = nat2mv(eta1, eta2)
%NAT2MV Convert Gaussian natural parameters to moments
%
% eta2 = -0.5 Sigma^{-1}
% eta1 = Sigma^{-1} mu

    Sigma = -0.5 * (eta2 \ eye(size(eta2)));
    mu = Sigma * eta1;
end