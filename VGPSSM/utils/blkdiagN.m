function B = blkdiagN(A, N)
%BLKDIAGN Repeat matrix A on block diagonal N times

    blocks = cell(1,N);
    for i = 1:N
        blocks{i} = A;
    end
    B = blkdiag(blocks{:});
end