function outVec = weightedFKNNR(matX, matY, inX, fW, K)
% weighted feature knn regression, fW is feature weights
% need a unit test

% check feature weights dimension, should be 1XN
assert(size(matX, 2) == size(fW, 2) && size(fW, 1) == 1);

% search for k nn
distFunc1 = @(x,Z,wt) sqrt((bsxfun(@minus,x,Z).^2)*wt');
distFunc = @(x1, x2) distFunc1( x1, x2, fW );
[indX, DX] = knnsearch(matX, inX, 'K', K, 'Distance', distFunc);

eDX = exp(-DX); % in case of 0 dist 
weights = eDX ./ repmat(sum(eDX, 2), [1, size(eDX, 2)]);
outVec = sum( matY(indX) .* weights , 2);
