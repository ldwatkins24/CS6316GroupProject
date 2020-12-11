#%% Packages
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate

import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve

import seaborn as sns


#%% Load Data
# Note: Column header added to data for dataframe

data1 = pd.read_csv("project3_dataset1csv.csv")
data2 = pd.read_csv("project3_dataset2csv.csv")

#%% Convert categorical data into binary data for data2
# Note: 1 == Present, 0 == Absent

class_array = []
for index, row in data2.iterrows():
    if row['Col 5'] == 'Present':
        class_array.append(1)
    elif row['Col 5'] == 'Absent':
        class_array.append(0)
    else:
        print("ERROR, encountered wrong class, please filter")
data2['Col 5'] = class_array

#%% Organize data into X and Y

X1 = data1.drop('Col 31', axis=1)
Y1 = data1['Col 31']

X2 = data2.drop('Col 10', axis=1)
Y2 = data2['Col 10']

#%% Randomly withold data for testing
# Note: Need to ensure same proportion of positives and negatives in both sets

# Self-check for number of positives and negatives

checkNumPositives1 = []
checkNumNegatives1 = []
checkNumPositives2 = []
checkNumNegatives2 = []

for value in Y1:
    if value == 1:
        checkNumPositives1.append(value)
    else: checkNumNegatives1.append(value)
    
for value in Y2:
    if value == 1:
        checkNumPositives2.append(value)
    else: checkNumNegatives2.append(value)

print("data1 has %d positives" %(len(checkNumPositives1)))
print("data1 has %d negatives" %(len(checkNumNegatives1)))
print("data2 has %d positives" %(len(checkNumPositives2)))
print("data2 has %d negatives" %(len(checkNumNegatives2)))

# Calculate number of positives and negatives to be randomly removed from training data
proportionPositives1 = Y1.sum()/len(Y1)
proportionPositives2 = Y2.sum()/len(Y2)
testSetProportion = 0.2

numPositives1 = int(testSetProportion*proportionPositives1*len(Y1))
numNegatives1 = int(testSetProportion*len(Y1))-numPositives1
numPositives2 = int(testSetProportion*proportionPositives2*len(Y2))
numNegatives2 = int(testSetProportion*len(Y2))-numPositives2

print("I need %d positives and %d negatives from data1 for 20 percent test data" %(numPositives1, numNegatives1))
print("I need %d positives and %d negatives from data2 for 20 percent test data" %(numPositives2, numNegatives2))

#%% ## Select a proportion of the right number of positive/negative for test, take remainder for training

#%% Find indexes for all positive and negative data

index_all1 = data1.index

condition1Pos = data1['Col 31']==1
indexesPositive1 = index_all1[condition1Pos].tolist()

index_all2 = data2.index

condition2Pos = data2['Col 10']==1
indexesPositive2 = index_all2[condition2Pos].tolist()


condition1Neg = data1['Col 31']==0
indexesNegative1 = index_all1[condition1Neg].tolist()

condition2Neg = data2['Col 10']==0
indexesNegative2 = index_all2[condition2Neg].tolist()

#%% Randomly grab appropriate number of positives and negatives

randomPosSample1 = random.sample(indexesPositive1, numPositives1)
randomNegSample1 = random.sample(indexesNegative1, numNegatives1)
combinedTestSetIndexes1 = randomPosSample1+randomNegSample1

randomPosSample2 = random.sample(indexesPositive2, numPositives2)
randomNegSample2 = random.sample(indexesNegative2, numNegatives2)
combinedTestSetIndexes2 = randomPosSample2+randomNegSample2

#%% Form test and training dataframes; Split into X and Y

testSet1 = data1[data1.index.isin(combinedTestSetIndexes1)]
trainingSet1 = data1[~data1.index.isin(combinedTestSetIndexes1)]

testSet2 = data2[data2.index.isin(combinedTestSetIndexes2)]
trainingSet2 = data2[~data2.index.isin(combinedTestSetIndexes2)]

testX1 = testSet1.drop('Col 31', axis=1)
testY1 = testSet1['Col 31']
testX2 = testSet2.drop('Col 10', axis=1)
testY2 = testSet2['Col 10']

trainX1 = trainingSet1.drop('Col 31', axis=1)
trainY1 = trainingSet1['Col 31']
trainX2 = trainingSet2.drop('Col 10', axis=1)
trainY2 = trainingSet2['Col 10']


def createValidationCurves(model,X,y,param,param_range,cv_n, score):
    #Create Validation Curves

    train_scores, valid_scores = validation_curve(model, X, y, param, param_range, scoring = score, cv=cv_n, n_jobs = -1)
    train_scores_mean = np.mean(train_scores,axis=1)
    valid_scores_mean = np.mean(valid_scores,axis=1)

    train_scores_std = np.std(train_scores,axis=1)
    valid_scores_std = np.std(valid_scores,axis=1)

    
    return [train_scores_mean, valid_scores_mean, train_scores_std, valid_scores_std]

def printScores(scores1,scores2):
    acc1 = np.mean(scores1['test_acc'])
    prec1 = np.mean(scores1['test_prec'])
    rec1 = np.mean(scores1['test_rec'])
    AUCROC1 = np.mean(scores1['test_AUC'])
    f1_1 = np.mean(scores1['test_f1'])

    acc2 = np.mean(scores2['test_acc'])
    prec2 = np.mean(scores2['test_prec'])
    rec2 = np.mean(scores2['test_rec'])
    AUCROC2 = np.mean(scores2['test_AUC'])
    f1_2 = np.mean(scores2['test_f1'])

    print("\n")
    print("Accuracy Dataset1: %f" % (acc1))
    print("Precision Dataset1: %f" % (prec1))
    print("Recall Dataset1: %f" % (rec1))
    print("AUC-ROC Dataset1: %f" % (AUCROC1))
    print('F-1 Dataset1: %f' % (f1_1))

    print("\n")
    print("Accuracy Dataset2: %f" % (acc2))
    print("Precision Dataset2: %f" % (prec2))
    print("Recall Dataset2: %f" % (rec2))
    print("AUC-ROC Dataset2: %f" % (AUCROC2))
    print('F-1 Dataset2: %f' % (f1_2))

def plotCurves(scores,ylow,yhigh,title, param, param_range, scoring, dataset):
    plt.title("DS%d: Validation Curve with %s" % (dataset, title))
    plt.xlabel(param)
    plt.ylabel("%s Score" % scoring)
    plt.ylim(ylow, yhigh)
    lw = 2
    plt.semilogx(param_range, scores[0], label="Training score",
                color="darkorange", lw=lw)
    plt.fill_between(param_range, scores[0]- scores[2],
                    scores[0] + scores[2], alpha=0.2,
                    color="darkorange", lw=lw)
    plt.semilogx(param_range, scores[1], label="Cross-validation score",
                color="navy", lw=lw)
    plt.fill_between(param_range, scores[1] - scores[3],
                    scores[1] + scores[3], alpha=0.2,
                    color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

#%% Create Models

#Logistic Regression
clf1_lr = LogisticRegression(penalty = 'none') # no regularization
clf1_lr.fit(trainX1,trainY1)

clf2_lr = LogisticRegression(penalty = 'none') # no regularization
clf2_lr.fit(trainX2,trainY2)

clf1_lrl2 = LogisticRegression(penalty = 'l2') # no regularization
clf1_lrl2.fit(trainX1,trainY1)

clf2_lrl2 = LogisticRegression(penalty = 'l2') # no regularization
clf2_lrl2.fit(trainX2,trainY2)

#KNN
clf1_knn = KNeighborsClassifier()
clf1_knn.fit(trainX1,trainY1)

clf2_knn = KNeighborsClassifier()
clf2_knn.fit(trainX2,trainY2)

#SVM
clf1_svm = svm.SVC(kernel='linear')
clf1_svm.fit(trainX1,trainY1)

clf2_svm = svm.SVC(kernel='linear')
clf2_svm.fit(trainX2,trainY2)

#Decision Tree
clf1_dt = tree.DecisionTreeClassifier()
clf1_dt.fit(trainX1,trainY1)

clf2_dt = tree.DecisionTreeClassifier()
clf2_dt.fit(trainX2,trainY2)

#Random Forest
clf1_rf = RandomForestClassifier()
clf1_rf.fit(trainX1,trainY1)

clf2_rf = RandomForestClassifier()
clf2_rf.fit(trainX2,trainY2)

#Boosting
clf1_gb = GradientBoostingClassifier()
clf1_gb.fit(trainX1,trainY1)

clf2_gb = GradientBoostingClassifier()
clf2_gb.fit(trainX2,trainY2)

#%% Create and Plot Validation Curves

#lr1_scores = createValidationCurves(clf1_lr,trainX1,trainY1,,,10,"accuracy")
#lr2_scores = createValidationCurves(clf2_lr,trainX1,trainY1,,,10,"accuracy")
#lr1_l2_scores = createValidationCurves(clf1_lrl2,trainX1,trainY1,,,10,"accuracy")
#lr2_l2_scores = createValidationCurves(clf2_lrl2,trainX1,trainY1,,,10,"accuracy")

#%% KNN Validation Curves

param = "n_neighbors"
param_range = np.arange(3,50,2)
scoring = {"accuracy","precision","recall","f1","roc_auc"}
for score_type in scoring:
    score1 = createValidationCurves(clf1_knn,trainX1,trainY1,param,param_range,10,score_type)
    score2 = createValidationCurves(clf2_knn,trainX2,trainY2,param,param_range,10,score_type)
    plotCurves(score1,0.8,1.1,"KNN",param,param_range,score_type,1)
    plotCurves(score2,0.0,1.1,"KNN",param,param_range,score_type,2)


#%% SVM Validation Curves

param = "gamma"
param_range = np.logspace(-6,1,10)
scoring = {"accuracy","precision","recall","f1","roc_auc"}
for score_type in scoring:
    score1 = createValidationCurves(clf1_svm,trainX1,trainY1,param,param_range,10,score_type)
    score2 = createValidationCurves(clf2_svm,trainX2,trainY2,param,param_range,10,score_type)
    plotCurves(score1,0.8,1.1,"SVM",param,param_range,score_type,1)
    plotCurves(score2,0.0,1.1,"SVM",param,param_range,score_type,2)

param = "C"
param_range = np.logspace(-1,2,10)
scoring = {"accuracy","precision","recall","f1","roc_auc"}
for score_type in scoring:
    score1 = createValidationCurves(clf1_svm,trainX1,trainY1,param,param_range,10,score_type)
    score2 = createValidationCurves(clf2_svm,trainX2,trainY2,param,param_range,10,score_type)
    plotCurves(score1,0.8,1.1,"SVM",param,param_range,score_type,1)
    plotCurves(score2,0.0,1.1,"SVM",param,param_range,score_type,2)


#%% Decision Tree Validation Curves

param = "max_depth"
param_range = np.arange(1,11)
scoring = {"accuracy","precision","recall","f1","roc_auc"}
for score_type in scoring:
    score1 = createValidationCurves(clf1_dt,trainX1,trainY1,param,param_range,10,score_type)
    score2 = createValidationCurves(clf2_dt,trainX2,trainY2,param,param_range,10,score_type)
    plotCurves(score1,0.8,1.1,"Decision Tree",param,param_range,score_type,1)
    plotCurves(score2,0.0,1.1,"Decision Tree",param,param_range,score_type,2)

#%% Random Forest Curves

param = "n_estimators"
param_range = np.arange(1,50,2)
scoring = {"accuracy","precision","recall","f1","roc_auc"}
for score_type in scoring:
    score1 = createValidationCurves(clf1_rf,trainX1,trainY1,param,param_range,10,score_type)
    score2 = createValidationCurves(clf2_rf,trainX2,trainY2,param,param_range,10,score_type)
    plotCurves(score1,0.8,1.1,"Random Forest",param,param_range,score_type,1)
    plotCurves(score2,0.0,1.1,"Random Forest",param,param_range,score_type,2)

param = "max_depth"
param_range = np.arange(1,11)
scoring = {"accuracy","precision","recall","f1","roc_auc"}
for score_type in scoring:
    score1 = createValidationCurves(clf1_rf,trainX1,trainY1,param,param_range,10,score_type)
    score2 = createValidationCurves(clf2_rf,trainX2,trainY2,param,param_range,10,score_type)
    plotCurves(score1,0.8,1.1,"Random Forest",param,param_range,score_type,1)
    plotCurves(score2,0.0,1.1,"Random Forest",param,param_range,score_type,2)


#%% Gradient Boosting Validation Curves

param = "n_estimators"
param_range = np.arange(1,50,2)
scoring = {"accuracy","precision","recall","f1","roc_auc"}
for score_type in scoring:
    score1 = createValidationCurves(clf1_gb,trainX1,trainY1,param,param_range,10,score_type)
    score2 = createValidationCurves(clf2_gb,trainX2,trainY2,param,param_range,10,score_type)
    plotCurves(score1,0.8,1.1,"Gradient Boosting",param,param_range,score_type,1)
    plotCurves(score2,0.0,1.1,"Gradient Boosting",param,param_range,score_type,2)


#%% Evaluate

# Cross-Validation Split
# cv = KFold(n_splits=10, random_state=1, shuffle=True)

scoring = {'acc': 'accuracy',
           'prec': 'precision',
           'rec': 'recall',
           'f1': 'f1',
           'AUC': 'roc_auc'}

#scores1 = evaluate_model(testX1, testY1, cv, clf1.fit(trainX1, trainY1))
#scores2 = evaluate_model(testX2, testY2, cv, clf1.fit(trainX2, trainY2))

lr_scores1 = cross_validate(clf1_lr, testX1, testY1, scoring=scoring, cv=10, n_jobs=-1)
lr_scores2 = cross_validate(clf2_lr, testX2, testY2, scoring=scoring, cv=10, n_jobs=-1)

lrl2_scores1 = cross_validate(clf1_lrl2, testX1, testY1, scoring=scoring, cv=10, n_jobs=-1)
lrl2_scores2 = cross_validate(clf2_lrl2, testX2, testY2, scoring=scoring, cv=10, n_jobs=-1)

svm_scores1 = cross_validate(clf1_svm, testX1, testY1, scoring=scoring, cv=10, n_jobs=-1)
svm_scores2 = cross_validate(clf2_svm, testX2, testY2, scoring=scoring, cv=10, n_jobs=-1)

knn_scores1 = cross_validate(clf1_knn, testX1, testY1, scoring=scoring, cv=10, n_jobs=-1)
knn_scores2 = cross_validate(clf2_knn, testX2, testY2, scoring=scoring, cv=10, n_jobs=-1)

dt_scores1 = cross_validate(clf1_dt, testX1, testY1, scoring=scoring, cv=10, n_jobs=-1)
dt_scores2 = cross_validate(clf2_dt, testX2, testY2, scoring=scoring, cv=10, n_jobs=-1)

rf_scores1 = cross_validate(clf1_rf, testX1, testY1, scoring=scoring, cv=10, n_jobs=-1)
rf_scores2 = cross_validate(clf2_rf, testX2, testY2, scoring=scoring, cv=10, n_jobs=-1)

gb_scores1 = cross_validate(clf1_gb, testX1, testY1, scoring=scoring, cv=10, n_jobs=-1)
gb_scores2 = cross_validate(clf2_gb, testX2, testY2, scoring=scoring, cv=10, n_jobs=-1)

printScores(lr_scores1,lr_scores2)
printScores(lrl2_scores1,lrl2_scores2)
printScores(knn_scores1,knn_scores2)
printScores(svm_scores1,svm_scores2)
printScores(dt_scores1,dt_scores2)
printScores(rf_scores1,rf_scores2)
printScores(gb_scores1,gb_scores2)


