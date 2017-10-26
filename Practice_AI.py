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

col_names = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
'Ticket', 'Fare', 'Cabin', 'Embarked']


class prepareData(object):
    """ Class so that anything I do to test, can be done to train easily"""

    def __init__(self,data,classType='test'):
        self.data = data
        self.classType = classType
        
        self.cleanAge()
        self.cleanSex()
        self.cleanEmbarked()
        self.cleanFare()

        if classType == 'train':
            self.defineTarget()
        self.defineFeatures()
        
    def cleanAge(self):
        self.data.Age = self.data.Age.fillna(self.data.Age.median())
        self.data['Child'] = float('NaN') # initalize with NaNs
        self.data.Child[self.data.Age >= 18] =0
        self.data.Child[self.data.Age < 18] = 1

    def cleanSex(self):
        self.data.Sex[self.data.Sex == 'male'] = 0
        self.data.Sex[self.data.Sex == 'female'] = 1
        
    def cleanEmbarked(self):
        self.data.Embarked = self.data.Embarked.fillna('S')
        self.data.Embarked[self.data.Embarked == 'S'] = 0 ## Why is this neecesary? Why not keep as str?
        self.data.Embarked[self.data.Embarked == 'C'] = 1
        self.data.Embarked[self.data.Embarked == 'Q'] = 2

    def cleanFare(self):
        self.data.Fare = self.data.Fare.fillna(self.data.Fare.median())
        
    def defineTarget(self):
        self.target = self.data.Survived.values

    def defineFeatures(self):
#        self.featureList = ['Pclass','Sex','Age','Fare','Child','SibSp','Parch']
        self.featureList = ['Pclass','Sex','Age','Fare','Child','SibSp']

        self.features = self.data[self.featureList].values
        if self.classType == 'train':
            print('feature list is '+str(self.featureList))

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC

#training,validating = train_test_split(train_data,test_size = .3,random_state = 3)

#train = prepareData(training,classType = 'train')
#validate = prepareData(validating,classType='train')


train = prepareData(train_data,classType = 'train')
tree_one = tree.DecisionTreeClassifier(max_depth = 7,min_samples_leaf = 4)
#tree_one = RFC(max_depth = 5,min_samples_leaf = 4)
tree_one = tree_one.fit(train.features,train.target)

print(tree_one.feature_importances_)
print(' ')
print(tree_one.score(train.features,train.target))

#print(tree_one.score(validate.features,validate.target))


#
test = prepareData(test_data)
test_prediction = tree_one.predict(test.features)

PassengerId =sp.array(test.data["PassengerId"]).astype(int)
my_solution = pd.DataFrame(test_prediction, PassengerId, columns = ["Survived"])
print(my_solution.shape) # Check data frame has 418 entries
my_solution.to_csv("my_solution.csv", index_label = ["PassengerId"])


"""
Score to beat is .77, which came from Tree- Pcalss, Sex Age Fare Child SibSp, local was .83

"""