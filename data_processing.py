# Pre-processing data to be suitable for our machine learning models
# Removed Null
# Replaced categorical columns with dummy variables

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

#print(test_set.head())
#print(train_set.head())

"""
By using sns.heatmap(train_set.isnull()) we figure out which columns have Null values.
Columns - 'Age' and 'Cabin' had a significant amount of null values and a few null values were also present in column
'Embarked'. We will replace the Null values in column 'Age' by finding the mean age for different PClass columns
by using the below command.
print(train_set.groupby('Pclass', as_index=False)['Age'].mean())
  Pclass        Age
0       1  38.233441
1       2  29.877630
2       3  25.140620
"""

def remove_null_age(cols):
    """ This function returns the mean age for the Pclass that the given person belongs to"""
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return 38

        elif Pclass == 2:
            return 30

        else:
            return 25

    else:
        return Age

""" Replacing the NUll Age values with mean age with respect to Pclass"""
train_set['Age'] = train_set[['Age','Pclass']].apply(remove_null_age,axis=1)

test_set['Age'] = test_set[['Age','Pclass']].apply(remove_null_age,axis=1)

"""
Dropping the whole column Cabin as it contains too many Null values and also it is not a great predictor
for our problem
"""
train_set.drop('Cabin',axis=1,inplace=True)
test_set.drop('Cabin',axis=1,inplace=True)

"""Dropping the remaing rows which contain Null values"""
train_set.dropna(inplace=True)
test_set.dropna(inplace=True)

"""
Replacing categorical columns with dummy values and then concatenating the dummy values and dropping the
original categorical columns
"""

#For training dataset
sex = pd.get_dummies(train_set['Sex'],drop_first=True)
embark = pd.get_dummies(train_set['Embarked'],drop_first=True)
train_set = pd.concat([train_set,sex,embark],axis=1)
train_set.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)

#For test dataset

sex = pd.get_dummies(test_set['Sex'],drop_first=True)
embark = pd.get_dummies(test_set['Embarked'],drop_first=True)
test_set = pd.concat([test_set,sex,embark],axis=1)
test_set.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)
#print(test_set.head())
#sns.heatmap(test_set.isnull())
plt.show()
