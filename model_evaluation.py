import pandas as pd
import numpy as np
import matplotlib as plt
import math
import random
from numpy import mean
from numpy import std
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#import seaborn as sns
# sns.set(style="white")


def evaluate_model(x, y, cv, model):
    # evaluate the model
    scoring = {'acc': 'accuracy',
               'prec': 'precision',
               'rec': 'recall',
               'f1': 'f1',
               'AUC': 'roc_auc'}

    scores = cross_validate(model, x, y, scoring=scoring, cv=cv, n_jobs=-1)
    # return scores
    return mean(scores['test_acc'])


# Load Data
dataset_1 = pd.read_csv('project3_dataset1.txt', sep='\t', header=None)
dataset_2 = pd.read_csv('project3_dataset2.txt', sep='\t', header=None)


# # Normalize and Transform Data
# Dataset 1
x_1 = dataset_1.values
min_max_scale = MinMaxScaler()
x_1_scaled = min_max_scale.fit_transform(x_1)
ds_1 = pd.DataFrame(x_1_scaled)
x_1 = ds_1.iloc[:, 0:-1].values
y_1 = ds_1.iloc[:, -1].values


# Dataset 2

dataset_2.loc[dataset_2[4] == 'Present', 4] = 1
dataset_2.loc[dataset_2[4] == 'Absent', 4] = 0

x_2 = dataset_2.values
x_2_scaled = min_max_scale.fit_transform(x_2)
ds_2 = pd.DataFrame(x_2_scaled)
x_2 = ds_2.iloc[:, 0:-1].values
y_2 = ds_2.iloc[:, -1].values

# Cross-Validation Split
cv = KFold(n_splits=10, random_state=1, shuffle=True)

# Create Models
# Logistic Regression
# log_model = LogisticRegression(penalty = 'none') # no regularization
# log_model_reg1 = LogisticRegression(penalty = 'l1') # with l1 regularization (doesn't work)
# log_model_reg2 = LogisticRegression(penalty = 'l2') # with l2 regularization

# K-Nearest Neighbors
knn_model = KNeighborsClassifier()

# Evaluate Models

scoring = {'acc': 'accuracy',
           'prec': 'precision',
           'rec': 'recall',
           'f1': 'f1',
           'AUC': 'roc_auc'}

# scores_1_log = cross_validate(log_model, x_1,y_1, scoring = scoring, cv=cv, n_jobs = -1)
# scores_1_l1 = cross_validate(log_model_reg1, x_1,y_1, scoring = scoring, cv=cv, n_jobs = -1)
# scores_1_l2 = cross_validate(log_model_reg2, x_1,y_1, scoring = scoring, cv=cv, n_jobs = -1)

# scores_2_log = cross_validate(log_model, x_2,y_2, scoring = scoring, cv=cv, n_jobs = -1)
# scores_2_l1 = cross_validate(log_model_reg1, x_2,y_2, scoring = scoring, cv=cv, n_jobs = -1)
# scores_2_l2 = cross_validate(log_model_reg2, x_2,y_2, scoring = scoring, cv=cv, n_jobs = -1)

# scores_1_knn = cross_validate(knn_model, x_1, y_1, scoring = scoring, cv = cv, n_jobs = -1)
# scores_2_knn = cross_validate(knn_model, x_2, y_2, scoring = scoring, cv = cv, n_jobs = -1)

for k in range(1, 50):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = evaluate_model(x_1, y_1, cv, knn_model)
    print('%s: %f' % (k, scores))

# print(scoring)
# print(scores_1_log['test_acc'])

# print(scoring)
# print(scores_1_l1['test_acc'])

# print(scoring)
# print(scores_1_l2['test_acc'])

# print(scoring)
# print(scores_2_log['test_acc'])
# print(scoring)
# print(scores_2_l1['test_acc'])
# print(scoring)
# print(scores_2_l2['test_acc'])

# print(scoring)
# print(scores_1_knn['test_acc'])
# print(scoring)
# print(scores_2_knn['test_acc'])