function [XN, muX, stdX] = normalize_data(X)
    muX = mean(X,1);
    stdX = std(X,0,1);
    stdX(stdX < 1e-12) = 1.0;
    XN = (X - muX) ./ stdX;
end