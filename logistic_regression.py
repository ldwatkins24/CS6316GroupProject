import pandas as pd 
import numpy as np
import matplotlib as plt
import math
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold


# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax
 
# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
 

# Split dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = random.randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split


def confusion_matrix(actual, predicted):
	tp = 0
	fp = 0
	fn = 0
	tn = 0
	for i in range(len(actual)):
		if actual[i] == 1 and predicted[i] == 1:
			tp += 1
		elif actual[i] == 1 and predicted[i] == 0:
			fn += 1
		elif actual[i] == 0 and predicted[i] == 1:
			fp += 1
		elif actual[i] == 0 and predicted[i] == 0:
			tn += 1
	return [tp, fp, fn, tn]




# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = [0] * n_folds
	for i in range(len(folds)):
		train_set = list(folds)
		fold = train_set.pop(i)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		cf = confusion_matrix(actual, predicted)
		accuracy = (cf[0] + cf[3]) / (sum(cf)) * 100.0
		precision = (cf[0]) / (cf[0] + cf[1]) * 100.0
		recall = (cf[0]) / (cf[0] + cf[2]) * 100.0
		f_measure = (2*cf[0]) / (2*cf[0] + cf[2] + cf[1]) * 100.0
		scores[i] = [accuracy,precision,recall,f_measure]
	return scores


#Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        yhat += coefficients[i+1]*row[i]
    return 1.0 / (1.0 + math.exp(-yhat))

	
# Estimate logistic regression coefficients using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	# Loop over each epoch
	for epoch in range(n_epoch):
		sum_error = 0
		# Loop over each row in training data
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			sum_error += error**2
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return coef

#Logistic Regression
def logistic_regression(train, test, l_rate, n_epoch):
	predictions = list()
	coef = coefficients_sgd(train, l_rate, n_epoch)
	for row in test:
		yhat = predict(row, coef)
		yhat = round(yhat)
		predictions.append(yhat)
	return(predictions)

# Load Data
dataset_1 = pd.read_csv('project3_dataset1.txt',sep = '\t',header=None)
dataset_2 = pd.read_csv('project3_dataset2.txt',sep = '\t',header=None)

ds_1 = dataset_1.values

# Normalize Data
minmax_1 = dataset_minmax(ds_1)
normalize_dataset(ds_1,minmax_1)

# Implement Algorithm
n_folds = 10
l_rate = 0.1
n_epoch = 100
scores = evaluate_algorithm(ds_1,logistic_regression,n_folds,l_rate,n_epoch)
print('Scores: %s' % scores)
sum_metrics = np.sum(scores,0)
print('Mean Accuracy: %.3f%%' % (sum_metrics[0]/float(n_folds)))
print('Mean Precision: %.3f%%' % (sum_metrics[1]/float(n_folds)))
print('Mean Recall: %.3f%%' % (sum_metrics[2]/float(n_folds)))
print('Mean F-Measure: %.3f%%' % (sum_metrics[3]/float(n_folds)))

# Not sure if we can just use pre existing libraries. I think we can, but the above is for fun
# # Normalize Data
# x_1 = dataset_1.values
# min_max_scale = MinMaxScaler()
# x_1_scaled = min_max_scale.fit_transform(x_1)
# ds_1 = pd.DataFrame(x_1_scaled)

# # Create 10 Fold Split
# kf_1 = KFold(n_splits=10, shuffle = True, 1)

# result = next(kf_1.split(ds_1), None)

# train = ds_1.iloc[result[0]]
# test = ds_1.iloc[result[1]]

# Calculate coefficients
# dataset = [[2.7810836,2.550537003,0],
# 	[1.465489372,2.362125076,0],
# 	[3.396561688,4.400293529,0],
# 	[1.38807019,1.850220317,0],
# 	[3.06407232,3.005305973,0],
# 	[7.627531214,2.759262235,1],
# 	[5.332441248,2.088626775,1],
# 	[6.922596716,1.77106367,1],
# 	[8.675418651,-0.242068655,1],
# 	[7.673756466,3.508563011,1]]
# l_rate = 0.3
# n_epoch = 100
# coef = coefficients_sgd(dataset, l_rate, n_epoch)
# print(coef)


