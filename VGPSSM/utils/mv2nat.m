function [eta1, eta2] = mv2nat(mu, Sigma)
%MV2NAT Convert Gaussian moments to natural parameters
%
% q(x) = N(mu, Sigma)
% eta1 = Sigma^{-1} mu
% eta2 = -0.5 Sigma^{-1}

    if size(mu,2) ~= 1
        mu = mu(:);
    end

    Sinv = Sigma \ eye(size(Sigma));
    eta1 = Sinv * mu;
    eta2 = -0.5 * Sinv;
end