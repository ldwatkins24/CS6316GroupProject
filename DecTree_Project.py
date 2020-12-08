#%% Packages
import pandas as pd
import numpy as np
import random
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate



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

#%% Use sklearn.tree

clf1 = tree.DecisionTreeClassifier()
clf1.fit(trainX1,trainY1)

clf2 = tree.DecisionTreeClassifier()
clf2.fit(trainX2,trainY2)

#%% Plot Tree

tree.plot_tree(clf1)
tree.plot_tree(clf2)

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

scores1 = cross_validate(clf1.fit(trainX1, trainY1), testX1, testY1, scoring=scoring, cv=10, n_jobs=-1)
scores2 = cross_validate(clf2.fit(trainX2, trainY2), testX2, testY2, scoring=scoring, cv=10, n_jobs=-1)

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