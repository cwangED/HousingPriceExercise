function outVec = weightedKNNR(matX, matY, inX, K, distFunc)
% weighted knn regression
% need a unit test

% search for k nn
if nargin <5
    [indX, DX] = knnsearch(matX, inX, 'K', K);
else
    [indX, DX] = knnsearch(matX, inX, 'K', K, 'Distance', distFunc);
end

eDX = exp(-DX); % in case of 0 dist 
weights = eDX ./ repmat(sum(eDX, 2), [1, size(eDX, 2)]);
outVec = sum( matY(indX).*weights, 2);