#Support Vector Machine -score 0.82 precision

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

"""Importing the data pre processing module to preprocess the data"""
from data_processing import *

"""Setting X and y training variables for Decision Tree model"""
X = train_set.drop('Survived',axis = 1)
y = train_set['Survived']

"""
Splitting the dataset into training and test dataset in order calculate accuracy of our model
"""
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.30,random_state=101)

"""
Initialising the Decision Tree classifier
"""
svm = SVC()

"""
Training and predicting y_test values for our dataset
"""
svm.fit(X_train,y_train)
predictions = svm.predict(X_test)


"""
Checking Score of the decision tree model
"""
print("Support Vector Machine: \n",classification_report(y_test,predictions))


"""
The SVM takes default values for C and gamme which may not neccessarily produce the best prediction.
We can use gridsearch to try and find the best combination of parameters. We pass a dictionary with
parameter as keys and a list of values to check for that parameter.  
"""

param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}

grid = GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(X_train,y_train)

print(grid.best_params_)
# output - {'gamma': 0.001, 'C': 1000}
print(grid.best_estimator_)
"""
output :-
SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
"""

grid_pred = grid.predict(X_test)
print("SVM improved:\n",classification_report(y_test,grid_pred))
# Score improved by 16%
