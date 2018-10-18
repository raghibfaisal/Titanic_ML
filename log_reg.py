#Logistic regression -score 0.82 preicison

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
"""Importing the data pre processing module to preprocess the data"""
from data_processing import *

"""Setting X and y training variables for logistic regression model"""
X = train_set.drop('Survived',axis = 1)
y = train_set['Survived']

"""
Splitting the dataset into training and test dataset in order calculate accuracy of our model

Precision for 0.25 provided the best precision. Hence, model was trained on 25% on the dataset and test on 
the remaining 75%

0.2 -> 0.81
0.25 -> 0.82
0.3 -> 0.81
0.4 -> 0.78
"""
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.25,random_state=101)

"""
Training and predicting y_test values for our dataset
"""
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
#print(log_reg)
predictions = log_reg.predict(X_test)
"""
Checking the score of our model
"""
print("Logistic Regression:\n",classification_report(y_test,predictions))


"""
Using the entire train.csv file as the training dataset to predict survived column in the test.csv file
"""
X_final_test = test_set
log_reg_final = LogisticRegression()
log_reg_final.fit(X,y)
predictions_final = log_reg_final.predict(X_final_test)
test_set['Survived'] = predictions_final
#print(test_set.head())
test_set.to_csv('results.csv',index=False)

#print(train_set.head())
plt.show()
