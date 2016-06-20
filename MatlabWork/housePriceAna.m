%% Chengjia Wang Housing Price Exercise
%
% *This is a regression problem:*
% This project aims to predict the housing prices using UK governments land
% registry data. Personally, I recommand to use *Markov Hidden Model* which
% treat the data as a time-serie for optimal and through solution (housing
% price is more suitable to solve with consideration of trade time).
% However, *due to the limit of features and time* in this exercise,
% it may make ignorable differences using complicated or mixed prediction models,
% such as, boosting, Bayesian weighting, etc., or
% use complicated and state-of-art prediction models such as the ones mentioned in:
%
%
% # http://yann.lecun.com/exdb/publis/pdf/caplin-ssrn-08.pdf
% # http://www.hindawi.com/journals/aaa/2014/648047/
% # http://www.doc.ic.ac.uk/~mpd37/theses/2015_beng_aaron-ng.pdf
%
% Thus in this exercise, I will just use some existing models based on my
% knowledge directly on the cleaned and sampled data for prediction
% purposes with little innovation. These simple models include:
%
% * KNN and Maximum likelihood Kernel Smooth Regression (MLR)
% * Support Vector Regression (SVR)
% * Random Forrest (RF)
%
% And the solution is given in (*can pick the one you prefer*):
%
%
% * Matlab -> LR (Due to best support to matrix algebra and fast prototyping property)
% * R -> SVR (Free and clean)
% * Python -> RF (Free and wide range of ML lib)
%
% *Coding style personal option:* OOP is not a good way for small-code fast
% prototyping, but a good dilerable coding style which need careful design
%
% *Questions in the exercise description is answered in the code*
%
% *Claim:* All the method is implemented by myself, algorithms may refer to
% Bishop's book and some online source.

%% Preprocessing: sample and clean data
%
%
% * Sampled year prices for quick and easy implementation (e.g., to solve
% optimization problem using normal equation)
% * Encode different types of data all to numeric type
% * Details in _Preprocessing.m_
% Preprocessing

%% Initialization
% set paths, load and rescale data
clear; close all; fclose all;

trainStruc.Path = '/home/cwang/Desktop/amazonInterview/trainFile.csv';
testStruc.Path = '/home/cwang/Desktop/amazonInterview/testFile.csv';

trainStruc.Data = sortrows(csvread(trainStruc.Path), 2);
testStruc.Data = sortrows(csvread(testStruc.Path), 2);

% column-wise max-min normalization for data
trainStruc.nomData = mmNormaliz2D(trainStruc.Data);
testStruc.nomData = mmNormaliz2D(testStruc.Data);
% here the outliers should be found and elimitated

%% Simplest solution: KNN
% If time is strictly not allowed to use as a feature, all inputs are
% discrete variables. Then the simplest naive solution is simply K-nearest
% neighbour (KNN) and kernal density estimation (KDE).
%
% Property type is imcomparable qualitative data. Thus the distance
% function can be defined as logical value

trainStruc.X = trainStruc.Data(:, 3:5);
trainStruc.Y = trainStruc.Data(:, 1);

testStruc.X = testStruc.Data(:, 3:5);
testStruc.Y = testStruc.Data(:, 1);

% define qualitative distance function
knnStruc.distFunc = @(x1, x2) int8(bsxfun(@eq,x1,x2));

%%
% 1. Dicectly use matlab Knn class (matlab only provided classification
% function, thus the error will be large)
warning off;
for nK = 1:20
    knnStruc.md = fitcknn(trainStruc.X, trainStruc.Y, 'NumNeighbors', nK);
    knnStruc.Y = knnStruc.md.predict(testStruc.X);
    knnStruc.error(nK) = sum( sqrt( (knnStruc.Y - testStruc.Y).^2 ) )./...
        size(knnStruc.Y, 1);
    knnStruc.mdCross = knnStruc.md.crossval; % 10 fold cross validation
%     disp([num2str(nK), ':']);
%     disp(['kfoldloss: ', num2str(knnStruc.mdCross.kfoldLoss)]);
%     disp(['error: ', num2str(knnStruc.error(nK))]);
end
warning on;
disp(['Minmum Matlab Knn RMS error:', num2str(min(knnStruc.error))])
% as we can see, treat it as a classification process, extremely bad performance,
% kfoldloss over 0.99
%%
% 2. weighted Knn Regression
% the final prediction is weighted by both the distance and importance of
% the property (need to implement cross validation and optimization)

% First, decide the K using a simple way. Actually can be decided
% simultaneously with the weights

tempError = zeros(200, 1);
for nK = 1:200 % candidate Ks
    predFun = @(xTrain, yTrain, xTest) weightedKNNR( xTrain, yTrain, xTest,...
        nK);
    tempError(nK) = crossval('mse', trainStruc.X, trainStruc.Y, 'Predfun', predFun);
    
%         disp(nK)
end

%%
% find the first smallest error
tempInd = 1:size(tempError, 1);
knnStruc.K = min( tempInd( tempError ==  min(tempError) ) );
disp(['optimal Kis: ', num2str(knnStruc.K)]) % K is 161 in this case, which is a little bit big

%%
% but when we plot the errors it shows that when it's bigger than 50, the
% error doesn't change much. (To find it automatic, use the gradient(x) >
% thres method.)
figure, plot(tempError, 'LineWidth', 3);
snapnow;
knnStruc.K = 50;
knnStruc.IniError = sqrt(tempError(50)); % it ws mean square error, now RSE

%%
% Obtain weitht of features, taking the cross validation error as cost
% function. To avoid calculation of gradients, we can use particle swarm
% optimization (PSO), which is a gradient independent global method. I
% personally invinted a UKF-PSO methods which guide the PSO evolutionary
% process using unscented Kalman filter. The code will be published online
% soon after a journal paper is submitted.

knnStruc.fw = 1/size(trainStruc.X, 2) .* ones( 1, (size(trainStruc.X, 2)) );
knnStruc.CostFunc = @(fW) crossErrorKnn(trainStruc.X, trainStruc.Y, fW, knnStruc.K);
knnStruc.optOptions = optimoptions('particleswarm','SwarmSize',30,'HybridFcn',@fmincon);
% knnStruc.optOptions.Display = 'iter';
% using a trick here to reduce the searching space to 2D
[knnStruc.fW, tempWError] = particleswarm(knnStruc.CostFunc,3, [0 0 0]', [1 1 1]', knnStruc.optOptions);
knnStruc.fW = knnStruc.fW./sum(knnStruc.fW);

% Now the new cross-valid error is:
knnStruc.cvError = sqrt(tempWError);

% Calculate training error for bias-variance trade-off analysis
knnStruc.trainY = weightedFKNNR(trainStruc.X, trainStruc.Y, trainStruc.X,...
    knnStruc.fW, knnStruc.K);

knnStruc.trainError = sqrt(sum((trainStruc.Y - knnStruc.trainY).^2)/size(trainStruc.Y, 1));
%% Testing analysis Knn
% finally test the hypothesis "weighted feature KNN Regresser" in
% weightedFKNNR.m
knnStruc.Y = weightedFKNNR(trainStruc.X, trainStruc.Y, testStruc.X,...
    knnStruc.fW, knnStruc.K);
knnStruc.testError = sqrt(sum((testStruc.Y - knnStruc.Y).^2)/size(testStruc.Y, 1));

disp('Feature weighted Knn: ');
disp('----- Train Result-------');
disp([num2str(numel(trainStruc.Y)), ' training samples: ']);
disp(['Cross Validation Error: ', num2str(knnStruc.cvError)]);
disp(['Training Error: ', num2str(knnStruc.trainError)]);
disp('----- Test Result -------');
disp(['Testing Error: ', num2str(knnStruc.testError)]);
% Result even worse than directly use Knn: need tuning

%% Considering the data as time series
% Housing price data is a time series data, which is the main reason real
% estate become a value maintaining method in many countries. If we can
% treat the data as time series, a much wider range of algorithm can be
% used, altough Knn is still applicable (knn is expansive to compute, for
% saving time, the following code is commented).
%
%
% To apply Knn we just need to follow the same procedure shown above with
% proper initializations
%%
% * initialization
%

%%
%   % normalize time to number of days
%   testStruc.nomData(:, 2) = trainStruc.nomData(:, 2) - datenum('1995-1-1');
%   testStruc.nomData(:, 2) = trainStruc.nomData(:, 2) - datenum('1995-1-1');
%
%   trainStruc.X = trainStruc.Data(:, 2:5);
%   trainStruc.Y = trainStruc.Data(:, 1);
%   testStruc.X = testStruc.Data(:, 3:5);
%   testStruc.Y = testStruc.Data(:, 1);
%%
% * get K
%
%%
%   knnStruc.K = 50; % just being lazy here to save time, should do cross validation

%%
% * Train
%
%%
%   knnStruc.fW = 1/3 .* ones( 1, size(trainStruc.X, 2) );
%   knnStruc.CostFunc = @(fW) crossErrorKnn(trainStruc.X, trainStruc.Y, fW, knnStruc.K);
%   knnStruc.optOptions = optimoptions('particleswarm','SwarmSize',30,'HybridFcn',@fmincon);
% %   knnStruc.optOptions.Display = 'iter';
%   [knnStruc.fW, tempWError] = particleswarm(knnStruc.CostFunc,3, [0 0 0]', [1 1 1]', knnStruc.optOptions);
%   knnStruc.fW = knnStruc.fW./sum(knnStruc.fW);
%   knnStruc.trainY = weightedFKNNR(trainStruc.X, trainStruc.Y, trainStruc.X,...
%                 knnStruc.fW, knnStruc.K);
%   knnStruc.trainError = sqrt(sum((trainStruc.Y - knnStruc.trainY).^2)/size(trainStruc.Y, 1));

%%
% * Test
%
%%
%   knnStruc.Y = weightedFKNNR(trainStruc.X, trainStruc.Y, testStruc.X,...
%                   knnStruc.fW, knnStruc.K);
%   knnStruc.testError = sqrt(sum((testStruc.Y - knnStruc.Y).^2)/size(testStruc.Y, 1));
%%
% * Evaluation
%
% We should definitely give
%
% # Plots of the testing errors verses different parameter settings.
% # Perform *Bias-Variance* analysis for underfitting and overfitting
% # Comparison, Etc.
%

%% Functions used in Knn Estimater
% * weightedKNNR.m (used to train Knn for optimal K)
%
% <include>weightedKNNR.m</include>
%
% * crossErrorKnn.m (cross validation error for Knn)
%
% <include>crossErrorKnn.m</include>
%
% * weightedFKNNR.m (the final prediction model: feature weighted Knn)
%
% <include>weightedFKNNR.m</include>
%
% * unit tests
%
% <include>knnTests.m</include>
%

%% Kernal based Non-parametric Regression
% Again, treating the data as time series, a regression model can be used
% to solve this problem. Unlike the parametric models which requires to set
% parameters for kernal basis and regularization weights. Non parametric
% model can be used, e.g., Nadaraya-Watson kernel regression (simplest one).
%

%%
% 1. reload data and get rid of outliers

trainStruc.Path = '/home/cwang/Desktop/amazonInterview/trainFile.csv';
testStruc.Path = '/home/cwang/Desktop/amazonInterview/testFile.csv';

trainStruc.Data = sortrows(csvread(trainStruc.Path), 2);
testStruc.Data = sortrows(csvread(testStruc.Path), 2);
trainStruc.Y = trainStruc.Data(:, 1);

% Outliers elimination: 
% simplified version: get rid of too high prices
% robust solution should use linear regression + RANSAC (leave for time limit)
outlierInd = (trainStruc.Y >= mean(trainStruc.Y) + 2*std(trainStruc.Y));
trainStruc.Data = trainStruc.Data(outlierInd~=1, :);

% normalize time to number of days
trainStruc.Data(:, 2) = trainStruc.Data(:, 2) - datenum('1995-1-1');
testStruc.Data(:, 2) = testStruc.Data(:, 2) - datenum('1995-1-1');

trainStruc.X = trainStruc.Data(:, 2:5);
trainStruc.Y = trainStruc.Data(:, 1);

testStruc.X = testStruc.Data(:, 2:5);
testStruc.Y = testStruc.Data(:, 1);

%%
% 2. Kernal smooth ML regression (Gaussian Kernel)
%
% A simple implementation can be found at: Mathwork file exchange. Here's
% the simplest implementation: fit a time curve in every case (e.g., a
% curve for property that in london, particular type and lease duration).
%
% We first try to fit a curve for all train data, later fit a curve for
% each case using 3rd party tools.

% Fit a GP curve (Matlab internal function):
tic;
MLRStruc.GPModel = fitrgp(trainStruc.X, trainStruc.Y, 'KernelFunction', ...
    'squaredexponential');
toc;

% plot the fitted curve with training data
MLRStruc.GPTrainY = resubPredict(MLRStruc.GPModel);
figure, plot(trainStruc.X(:, 1), trainStruc.Y, 'r*'), hold on;
plot(trainStruc.X(:, 1), MLRStruc.GPTrainY, 'b--', 'LineWidth', 3), hold off;
snapnow;

%%
% 3. Test:

% make prediction
MLRStruc.GPTestY = predict(MLRStruc.GPModel, testStruc.X);

% RMS error
MLRStruc.GPError = sqrt(loss(MLRStruc.GPModel, testStruc.X, testStruc.Y));
disp('============================================================')
disp('Kernel based regress:')
disp(['Test Error: ', num2str(MLRStruc.GPError)]);

%% Using 3rd party packages (PRTools)
% *PRTools* and *Vlfeat* are widely used Matlab lib which contains popular
% machine learning and pattern recognition algorithms and tools. Both
% requires a little bit learning effort, but once mastered this two
% package, the Matlab code can be much shorter and descent. 
%
% Here we fit a curve for each case (simply linear regression)
%
% install prtools and import
% addpath './prtools'
addpath(genpath('./prtools'))
% addpath(genpath('./prtools4.2.5'))
% addpath('./@prdataset')



% encode each cases
PR = [1 2 3 4]; % property type
LN = [0 1]; % whether in london
LD = [1 2]; % lease duration
MLRStruc.codeBook = zeros(4*2*2, 3); % 4 for property types, 2 for location, 2 for duration
for i = 1:4
    for j = 1:2
        for k = 1:2
            MLRStruc.codeBook((i-1)*4+(j-1)*2 + k, :) = ...
                [PR(i), LN(j), LD(k)]';
        end
    end
end
%%
% Train: fit a curve for each case: e.g: property type = 1, not in london,
% lease duration type = 1

prwarning off;
trainStruc.PRDataset = [];
trainStruc.PRRegressor = [];
figure,
for nC = 1:size(MLRStruc.codeBook, 1)
    tempInd = (trainStruc.X(:, 2) == MLRStruc.codeBook(nC, 1))...
        .* (trainStruc.X(:, 3) == MLRStruc.codeBook(nC, 2))...
        .* (trainStruc.X(:, 4) == MLRStruc.codeBook(nC, 3));
    x = trainStruc.X(tempInd==1, 1); % time serie
    y = trainStruc.Y(tempInd==1);

    % Make a Prtools training regression dataset
    trainStruc.PRDataset{nC} = gendatr(x, y);
    
    % Train a regressor (kernal smooth)
    trainStruc.PRRegressor{nC} = trainStruc.PRDataset{nC}*linearr([], 1);
    
    % Plot the regression result
    subplot(4, 4, nC), plot(x, y, 'r*'), hold on;
    title(['PR: ', MLRStruc.codeBook(nC, 1), ...
            ', LN: ', MLRStruc.codeBook(nC, 2), ...
            ', LD: ', MLRStruc.codeBook(nC, 3)]);
    plotr(trainStruc.PRRegressor{nC}, 'b--');
    drawnow;
end
snapnow;

%% 
% Test: 
disp('================================================================')
disp('Linear regression for each case: ');
testStruc.PRDataset = [];
testStruc.MSE = [];
for nC = 1:size(MLRStruc.codeBook, 1)
    tempInd = (testStruc.X(:, 2) == MLRStruc.codeBook(nC, 1))...
        .* (testStruc.X(:, 3) == MLRStruc.codeBook(nC, 2))...
        .* (testStruc.X(:, 4) == MLRStruc.codeBook(nC, 3));
    x = testStruc.X(tempInd==1, 1); % time serie
    y = testStruc.Y(tempInd==1);

    % Make a Prtools test regression dataset
    testStruc.PRDataset{nC} = gendatr(x, y);
    
    % Testing the ( mean square error)
    testStruc.MSE{nC} = testStruc.PRDataset{nC}*trainStruc.PRRegressor{nC}*testr;
    try
     % display RMS error
     disp(['Error for PR:', num2str(MLRStruc.codeBook(nC, 1)), ...
         ', LN:', num2str(MLRStruc.codeBook(nC, 2)), ', LD:', ...
         num2str(MLRStruc.codeBook(nC, 3)), ': ', num2str(sqrt(testStruc.MSE{nC}))]);
    catch
     disp(['Error for PR:', num2str(MLRStruc.codeBook(nC, 1)), ...
         ', LN:', num2str(MLRStruc.codeBook(nC, 2)), ', LD:', ...
         num2str(MLRStruc.codeBook(nC, 3)), ': ', 'Empty class!!']);
    end
end 
prwarning on;
%% Matlab Summary
% We used Knn classification, a feature-weighted Knn regressor (just for fun), a Kernel
% based regression, and a linear regression model. A interesting discovery
% is that considering the data as time series even resulted to larger
% error. After we go deeper by fitting a curve for each case. It seems that
% there are for classes missing data. Further experiment should be
% conducted with larger sampled dataset, more carefully designed outlier
% elimination, and some parametric tuning steps. Anyway, the feature is
% limited. For housing prices, important factors include more precise
% location, date, etc.