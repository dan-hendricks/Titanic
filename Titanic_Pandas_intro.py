# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:16:35 2016

@author: DHendricks
"""

from matplotlib import pyplot as plt
import pandas as pd
import scipy as sp

train_url="http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
test_url="http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"

test = pd.read_csv(test_url)
train = pd.read_csv(train_url)

##print(train.describe())
#print(train["Survived"].value_counts())
## As proportions
#print(train["Survived"].value_counts(normalize=True))
#
## Males that survived vs males that passed away
#print(train["Survived"][train["Sex"]=='male'].value_counts())
#
## Females that survived vs Females that passed away
#print(train["Survived"][train["Sex"]=='female'].value_counts())
#
## Normalized male survival
#print(train["Survived"][train["Sex"]=='male'].value_counts(normalize=True))
## Normalized female survival
#print(train["Survived"][train["Sex"]=='female'].value_counts(normalize=True))
#


train["Child"] = float('NaN')
train["Child"][train["Age"]>=18] = 0
train["Child"][train["Age"]<18]=1
#print(train["Child"])

#print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))
#print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))

test_one = test
test_one["Survived"] = 0

# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
test_one["Survived"][test_one["Sex"]=="female"] = 1
print(test_one["Survived"])

