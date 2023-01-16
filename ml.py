import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

from sklearn import datasets

#make random dataset
X, y = make_regression(n_samples=5000, n_features=10)
#split in to train and test 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test
#Create a Gaussian Classifier
rfr=RandomForestRegressor()
#Train the model using the training sets 
rfr.fit(X_train,y_train)

y_pred=rfr.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
mse = metrics.mean_squared_error(ytest, ypred)