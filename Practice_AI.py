# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 19:06:18 2017

@author: DHendricks
"""
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd

from sklearn import tree
pd.options.mode.chained_assignment = None

train_url="http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
test_url="http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"

test_data = pd.read_csv(test_url)
train_data = pd.read_csv(train_url)

train_data.Age = train_data.Age.fillna(train_data.Age.median())


train_data['Child'] = float('NaN')  # For definitions, need to use brackets. ALl else, can use .
train_data.Child[train_data.Age >= 18] = 0
train_data.Child[train_data.Age < 18] = 1

train_data.Sex[train_data.Sex=='male'] = 0
train_data.Sex[train_data.Sex=='female'] = 1

train_data.Embarked = train_data.Embarked.fillna('S')
train_data.Embarked[train_data.Embarked == 'S'] = 0
train_data.Embarked[train_data.Embarked == 'C'] = 1
train_data.Embarked[train_data.Embarked == 'Q'] = 2

target = train_data.Survived.values

#features = train_data[['Pclass','Sex','Age','Fare']].values
features = train_data[['Pclass','Sex','Age','Fare','Child']].values

tree_one = tree.DecisionTreeClassifier()
tree_one = tree_one.fit(features,target)


#%%
#
##print('pclass, sex, age, fare')
print(tree_one.feature_importances_)
print(' ')
print(tree_one.score(features,target))
#
test_data.Fare = test_data.Fare.fillna(test_data.Fare.median())
#
test_data.Sex[test_data.Sex == 'male' ] = 0
test_data.Sex[test_data.Sex == 'female'] = 1
test_data.Age = test_data.Age.fillna(test_data.Age.median())
test_data['Child'] = float('NaN')
test_data.Child[test_data.Age < 18] = 1
test_data.Child[test_data.Age >= 18] = 0

#
#
#
#test_feature = test_data[['Pclass','Sex','Age','Fare']].values
#test_features = test_data[['Pclass','Sex','Age','Fare']].values
#
#test_prediction = tree_one.predict(test_features)

