function outMat = mmNormaliz2D( inMat )
%Performing column-wise max-min normalization for 2D matrix
minVec = min(inMat, [], 1);
maxVec = max(inMat, [], 1);
diffVec = repmat( maxVec - minVec, [size( inMat, 1 ) 1] );
outMat = ( inMat - repmat( minVec, [size(inMat, 1) 1] ) )./ diffVec;

% for the columns min = max, set to 1
outMat( isnan(outMat) ) = 1;
end

