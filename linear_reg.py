#Linear Regression - score 38%

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from data_processing import *

lm = LinearRegression()


"""Setting X and y training variables for linear regression model"""

X = train_set.drop('Survived',axis = 1)
y = train_set['Survived']

"""
Splitting the training data set into training and test data set to get an idea
of the accuracy of our model
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

"""
Training the linear regression model
"""
lm.fit(X_train,y_train)
#print(lm)
"""
Predicting the survived column for test set
"""
predictions = lm.predict(X_test)
#print(predictions)

"""
Calculating accuracy of the model 
"""
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
score = (lm.score(X_test,y_test))*100
print("Model accuracy is %g%%" %score)
X_test = test_set
#print(X_train)
lm.fit(X,y)
predictions_final = lm.predict(X_test)
#print(predictions_final)
#sns.pairplot(df)
plt.show()
