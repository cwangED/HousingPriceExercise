#!/usr/local/python
# Filename:housingPriceAna.py

'''Random Forrest Regression to predict housing price

Continue from the Rcode using SVM Regressor, in this python code, I use RFR.
Because time is extemely limited now, I won't use class for better oop.
Just function based implementations
'''

# import packages
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
# sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.svm import SVR


# 1. Read and clean data

# Set path
trainPath = "trainFile.csv"
testPath = "testFile.csv"

# load data

trainData = np.genfromtxt(trainPath, delimiter=',')
testData = np.genfromtxt(testPath, delimiter=',')

# normalize date (0-index) to number of days
testData[:, 1] = testData[:, 1] - trainData[:, 1].min()
trainData[:, 1] = trainData[:, 1] - trainData[:, 1].min()

# reorder data by date
trainData = trainData[np.argsort(trainData[:, 1]), :]
testData = testData[np.argsort(testData[:, 1]), :]

# split data
trainY = trainData[:, 0]
trainX = trainData[:, 1:]
testY = testData[:, 0]
testX = testData[:, 1:]

# Should eliminate outliers here

# 2. Fit the model

# creat RFR with 50 trees
RFRModel = RandomForestRegressor(random_state=0, n_estimators=50)

# cross valid
RFRScore = np.abs(cross_val_score(RFRModel, trainX, trainY).mean())
print("Training CV Error = %.2f" % RFRScore)

# fit model
RFRModel.fit(trainX, trainY)
RFRtrY = RFRModel.predict(trainX)

# training error
RFRtrError = sqrt(mean_squared_error(trainY, RFRtrY))
print("Training Error = %.2f" % RFRtrError)

# plot the fitted curves
plt.plot(trainX[:, 0], trainY, 'r*')
plt.plot(trainX[:, 0], RFRtrY, 'b--')
plt.ylim(0, 2e6)
plt.show()

# 3. Predict
RFRteY = RFRModel.predict(testX)
RFRteError = sqrt(mean_squared_error(testY, RFRteY))
print("Testing Error = %.2f" % RFRteError)

# 4. Should Tuning or bulid curves for each here

# 5. We can try other regressions using SVR

# Fit regression model
svrRBF = SVR(kernel='rbf', C=1e3, gamma=0.1) # radial basis function
svrLin = SVR(kernel='linear', C=1e3) # linear
svrPoly = SVR(kernel='poly', C=1e3, degree=2) # polynomial
svrRBFtrY = svrRBF.fit(trainX, trainY).predict(trainX)
svrLintrY = svrLin.fit(trainX, trainY).predict(trainX)
svrPolytrY = svrPoly.fit(trainX, trainY).predict(trainX)


# plts
plt.scatter(trainX, trainY, c='k', label='data')
plt.hold('on')
plt.plot(trainX, svrRBFtrY, c='g', label='RBF model')
plt.plot(trainX, svrLintrY, c='r', label='Linear model')
plt.plot(trainX, svrPolytrY, c='b', label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()

# test