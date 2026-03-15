function [naMat, nbMat, nkMat] = expand_narx_orders(na_in, nb_in, nk_in, ny, nu)

    % na: allow scalar, row vector, ny-by-ny matrix
    if isscalar(na_in)
        naMat = na_in * ones(ny, ny);
    elseif isvector(na_in) && numel(na_in) == ny
        naMat = repmat(reshape(na_in,1,[]), ny, 1);
    elseif isequal(size(na_in), [ny, ny])
        naMat = na_in;
    else
        error('Unsupported size for na.');
    end

    % nb: allow scalar, row vector of nu, or ny-by-nu matrix
    if isscalar(nb_in)
        nbMat = nb_in * ones(ny, nu);
    elseif isvector(nb_in) && numel(nb_in) == nu
        nbMat = repmat(reshape(nb_in,1,[]), ny, 1);
    elseif isequal(size(nb_in), [ny, nu])
        nbMat = nb_in;
    else
        error('Unsupported size for nb.');
    end

    % nk: allow scalar, row vector of nu, or ny-by-nu matrix
    if isscalar(nk_in)
        nkMat = nk_in * ones(ny, nu);
    elseif isvector(nk_in) && numel(nk_in) == nu
        nkMat = repmat(reshape(nk_in,1,[]), ny, 1);
    elseif isequal(size(nk_in), [ny, nu])
        nkMat = nk_in;
    else
        error('Unsupported size for nk.');
    end
end