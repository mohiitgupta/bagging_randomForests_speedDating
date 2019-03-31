import pandas as pd
import numpy as np
from trees import *
from preprocess_assg4 import preprocess
from scipy import stats
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = (7,7)


def plot_graph(x_axis, y_axis, se):
    plt.errorbar(x_axis, y_axis, yerr=se, fmt='-', capsize=4, capthick=2)


def split_cv(random_state, train_data, frac, num_folds):
    train_data_split = train_data.sample(frac=frac,random_state=random_state)
    train_data_split=np.array_split(train_data_split, num_folds)
    return train_data_split

def get_train_test_fold(train_data_split, test_idx):
    test_set = train_data_split[test_idx]
    train_set_list = []
    for i,x in enumerate(train_data_split):
        if i != test_idx:
            train_set_list.append(x)
    train_set = pd.concat(train_set_list)
    return train_set, test_set

def main():
    t0 = time.time()
    preprocess("dating-full.csv")
    trainingSet = pd.read_csv("trainingSet.csv")
    num_folds = 10
    train_data_split = split_cv(18, trainingSet, 1, num_folds)


    t_frac_list = [0.05,0.075,0.1,0.15,0.2]
    depth = 8
    num_trees = 30
    random_state_cv = 32
    accuracy_lists_models = []
    for i in range(3):
        accuracy_lists_models.append([])
    for t_frac in t_frac_list:
        for idx in range(num_folds):
            trainingSet, testSet = get_train_test_fold(train_data_split, idx)
            trainingSet = trainingSet.sample(frac=t_frac,random_state=random_state_cv)

            train_accuracy, test_accuracy = run_decisionTree(trainingSet, testSet, depth)
            accuracy_lists_models[0].append(test_accuracy)

            train_accuracy, test_accuracy = run_bagging(trainingSet, testSet, depth, num_trees)
            accuracy_lists_models[1].append(test_accuracy)

            train_accuracy, test_accuracy = run_randomForests(trainingSet, testSet, depth, num_trees)
            accuracy_lists_models[2].append(test_accuracy)
          
    avg_accuracy_models = []
    se_models = []
    for i in range(3):
        avg_accuracy_models.append([])
        se_models.append([])

    for i, t_frac in enumerate(t_frac_list):
        avg_accuracy_models[0].append(np.average(accuracy_lists_models[0][num_folds*i:num_folds*i+num_folds]))
        se_models[0].append(stats.sem(accuracy_lists_models[0][num_folds*i:num_folds*i+num_folds]))
        print ("t_frac", t_frac, "Model Decision Tree: test average accuracy", avg_accuracy_models[0][i], 
               "standard error:", se_models[0][i])

        avg_accuracy_models[1].append(np.average(accuracy_lists_models[1][num_folds*i:num_folds*i+num_folds]))
        se_models[1].append(stats.sem(accuracy_lists_models[1][num_folds*i:num_folds*i+num_folds]))
        print ("t_frac", t_frac, "Model Bagging: test average accuracy", avg_accuracy_models[1][i], 
               "standard error:", se_models[1][i])

        avg_accuracy_models[2].append(np.average(accuracy_lists_models[2][num_folds*i:num_folds*i+num_folds]))
        se_models[2].append(stats.sem(accuracy_lists_models[2][num_folds*i:num_folds*i+num_folds]))
        print ("t_frac", t_frac, "Model Random Forest: test average accuracy", avg_accuracy_models[2][i], 
               "standard error:", se_models[2][i])

    '''
    Plot learning Curves
    '''
    size_list = [t_frac*4680 for t_frac in t_frac_list]
    plot_graph(size_list, avg_accuracy_models[0], se_models[0])
    plot_graph(size_list, avg_accuracy_models[1], se_models[1])
    plot_graph(size_list, avg_accuracy_models[2], se_models[2])
    y_axis_label = "Average Model Accuracy"
    x_axis_label = "Size of Training Data"
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend(['Decision Tree', 'Bagged Trees','Random Forest'], loc='best')
        
    plt.title(y_axis_label + ' v/s ' + x_axis_label)
    # plt.show()
    plt.savefig("learning_curves_size_training_data",dpi=300)

    '''
    Do Hypothesis Testing
    '''
    for i,t_frac in enumerate(t_frac_list):

        print ("\nFor t_frac", t_frac, "run Paired t-test for Decision Tree and Bagging models")
        print ("\nNull Hypothesis H0: Decision Tree Accuracy = Bagging Accuracy")
        print ("Alternate Hypothesis H1: Decision Tree Accuracy != Bagging Accuracy")
        print (accuracy_lists_models[0][num_folds*i:num_folds*i+num_folds])
        print (accuracy_lists_models[1][num_folds*i:num_folds*i+num_folds])
        t_statistic, p_statistic = stats.ttest_rel(accuracy_lists_models[1][num_folds*i:num_folds*i+num_folds], accuracy_lists_models[0][num_folds*i:num_folds*i+num_folds])
        print ("Paired T-test Statistics are: t_statistic=", t_statistic, "pvalue=", p_statistic)
        if p_statistic < 0.05:
            print ("\nRejecting Null Hypothesis H0 since the pvalue is less than 0.05")
        else:
            print ("\nAccepting Null Hypothesis H0 since pvalue is greater than 0.05")

    t1 = time.time()
    total = t1-t0
    print ("total time taken for running code", total/60, "minutes")

if __name__ == '__main__':
    main()