function Xmean = stableMeanTrajectory(Xparticles)
%STABLEMEANTRAJECTORY Mean over particle dimension
%
% Input
%   Xparticles : [L x Dx x Np]
%
% Output
%   Xmean      : [L x Dx]

    if ndims(Xparticles) ~= 3
        error('stableMeanTrajectory: input must be [L x Dx x Np].');
    end

    Xmean = mean(Xparticles, 3);
end