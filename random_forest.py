#Random Forest model - score 0.79 precision

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

"""Importing the data pre processing module to preprocess the data"""
from data_processing import *


"""Setting X and y training variables for Decision Tree model"""
X = train_set.drop('Survived',axis = 1)
y = train_set['Survived']

"""
Splitting the dataset into training and test dataset in order calculate accuracy of our model
"""
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.40,random_state=101)

"""
Initialising the Decision Tree classifier
"""
dtree = DecisionTreeClassifier()

"""
Training and predicting y_test values for our dataset
"""
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)

"""
Checking Score of the decision tree model
"""
print("Decision Tree: \n",classification_report(y_test,predictions))

"""
Initialising the Random Forest classifier
"""
rfc = RandomForestClassifier(n_estimators=200)

"""
Training and predicting y_test values for our dataset by Random Forest
"""
rfc.fit(X_train,y_train)
#print(rfc)
rfc_predict = rfc.predict(X_test)

"""
Checking score of the Random Forest model
"""
print("Random Forest: \n",classification_report(y_test,rfc_predict))
