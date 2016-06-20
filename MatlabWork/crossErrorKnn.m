function errorOut = crossErrorKnn(trainX, trainY, fW, nK)

% cross validation error for feature weighted KNN regression
rng('default')
tempInd  = randi(size(trainY, 1), 1, 100);
predFun = @(xTrain, yTrain, xTest) weightedFKNNR( xTrain, yTrain, xTest, fW./sum(fW), nK);
errorOut = sqrt(crossval('mse', trainX(tempInd, :), trainY(tempInd, :), 'Predfun', predFun, 'kfold', 3));