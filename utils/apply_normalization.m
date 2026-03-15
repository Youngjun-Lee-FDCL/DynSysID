function XN = apply_normalization(X, muX, stdX)
    XN = (X - muX) ./ stdX;
end