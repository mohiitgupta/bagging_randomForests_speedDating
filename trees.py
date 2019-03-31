import pandas as pd
import numpy as np
import random
import math
import sys
import time

LABELS_LEN = 2

class Node:

    def __init__(self, attr, predicted_label):

        self.left = None
        self.right = None
        self.attr = attr
        self.predicted_label = predicted_label


    def PrintTree(self):
        
        if self.left:
            self.left.PrintTree()
        print (self.attr, " ", self.predicted_label)
        if self.right:
            self.right.PrintTree()

def find_gini(count_array, total_count):
    sum_prob = 0
    for count in count_array:
        prob = count/total_count
        sum_prob += pow(prob,2)
    return 1 - sum_prob

def calculate_gini(labels):
    # print ("label value counts", labels.value_counts())
    return find_gini(labels.value_counts(), len(labels))

def get_features_labels(dataset):
    features = dataset.drop(columns = ['decision'])
    labels = dataset['decision']
    return features, labels

def predict_label(labels):
    counts = labels.value_counts()
    predicted_label = counts.idxmax()
    confidence = 0
    if counts[predicted_label] == np.sum(counts):
        confidence = 1
    return predicted_label, confidence

def count_branch(attr, attr_value, dataframe):
    count_array = dataframe[dataframe[attr]==attr_value]['decision'].value_counts()
    return count_array, np.sum(count_array)

def calculate_gini_gain(attr, dataframe, gini_sample):
    count_array_left, total_eg_left = count_branch(attr, 0, dataframe)
    count_array_right, total_eg_right = count_branch(attr, 1, dataframe)
    # print (count_array_left)
    gini_left = find_gini(count_array_left, total_eg_left)
    # print (count_array_right)
    gini_right = find_gini(count_array_right, total_eg_right)
    
    gini_gain = gini_sample - gini_left*total_eg_left/len(dataframe) - gini_right*total_eg_right/len(dataframe)
    return gini_gain

def get_label_decision_tree(root, test_point):
    if root.attr is None:
        return root.predicted_label

    if test_point[root.attr] == 0:
        if root.left is not None:
            return get_label_decision_tree(root.left, test_point)
    if test_point[root.attr] == 1:
        if root.right is not None:
            return get_label_decision_tree(root.right, test_point)
    return root.predicted_label

def get_inference_single_tree(root, test_features, test_labels):
    predicted_labels = []
    for i in range(len(test_features)):
        predicted_label = get_label_decision_tree(root, test_features.iloc[i])
        predicted_labels.append(predicted_label)

    correct_labels = np.sum(predicted_labels == test_labels)
    accuracy = correct_labels/(1.0*len(test_labels))
    return predicted_labels, round(accuracy,2)

def get_inference_multiple_trees(baggedTrees, test_features, test_labels):
    predicted_labels = []
    for i in range(len(test_features)):
        predicted_label_array = []
        for j in range(len(baggedTrees)):
            predicted_label_i = get_label_decision_tree(baggedTrees[j], test_features.iloc[i])
            predicted_label_array.append(predicted_label_i)
        predicted_label = max(predicted_label_array, key = predicted_label_array.count)
        
        predicted_labels.append(predicted_label)
    correct_labels = np.sum(predicted_labels == test_labels)
    accuracy = correct_labels/(1.0*len(test_labels))
    return predicted_labels, round(accuracy,2)

def create_decision_tree(trainingSet, depth, is_random_forest, excluded_features, MAX_DEPTH):    
    predicted_label, confidence = predict_label(trainingSet['decision'])
    if (depth >= MAX_DEPTH) or (len(trainingSet) < 50) or (confidence == 1):
        return Node(None, predicted_label)

    # print ("\ndepth", depth, "length trainingset", len(trainingSet['decision']))
    gini_sample = calculate_gini(trainingSet['decision'])
    columns = trainingSet.columns
    # print ("gini sample", gini_sample, "column length", len(columns))
    
    '''
    sample sqrt(p) features incase of random forests
    '''
    if is_random_forest:
        num_samples = int(math.sqrt(len(columns)))
        columns = random.sample(set(columns), num_samples)
        
    max_gini_gain = -100
    max_split_attr = None
    for attr in columns:
        if attr != 'decision':
            gini_gain = calculate_gini_gain(attr, trainingSet, gini_sample)
            # if attr == "concerts":
            #     print (attr, gini_gain)
            if gini_gain >= max_gini_gain:
                max_gini_gain = gini_gain
                max_split_attr = attr

    # print ("max split attr", max_split_attr, "max gini gain", max_gini_gain)
    left_trainingSet = trainingSet[trainingSet[max_split_attr]==0]
    left_trainingSet = left_trainingSet.loc[:, left_trainingSet.columns != max_split_attr]

    right_trainingSet = trainingSet[trainingSet[max_split_attr]==1]
    right_trainingSet = right_trainingSet.loc[:, right_trainingSet.columns != max_split_attr]
    # print ("size of left child", len(left_trainingSet), "size of right child", len(right_trainingSet))
    node = Node(max_split_attr, predicted_label)

    if len(left_trainingSet) > 0:
        node.left = create_decision_tree(left_trainingSet, depth+1, is_random_forest, excluded_features, MAX_DEPTH)
    else:
        node.left = Node(None, predicted_label)
    
    if len(right_trainingSet) > 0:
        node.right = create_decision_tree(right_trainingSet, depth+1, is_random_forest, excluded_features, MAX_DEPTH)
    else:
        node.right = Node(None, predicted_label)
    return node

def create_bagged_trees(trainingSet, is_random_forest, MAX_DEPTH, num_trees):
    baggedTrees = []
    for i in range(num_trees):
        sampledSet= trainingSet.sample(frac=1,replace=True)
        root = create_decision_tree(sampledSet, 0, is_random_forest, {"decision"}, MAX_DEPTH)
        baggedTrees.append(root)
    return baggedTrees

def run_decisionTree(trainingSet, testSet, MAX_DEPTH):
    is_random_forest = False
    root = create_decision_tree(trainingSet, 0, is_random_forest, {"decision"}, MAX_DEPTH)

    training_features, training_labels = get_features_labels(trainingSet)
    test_features, test_labels = get_features_labels(testSet)
    test_predictions, test_accuracy = get_inference_single_tree(root, test_features, test_labels)
    train_predictions, train_accuracy = get_inference_single_tree(root, training_features, training_labels)
    return train_accuracy, test_accuracy

def decisionTree(trainingSet, testSet):
    return run_decisionTree(trainingSet, testSet, 8)

def run_bagging(trainingSet, testSet, MAX_DEPTH, num_trees):
    is_random_forest = False
    baggedTrees = create_bagged_trees(trainingSet, is_random_forest, MAX_DEPTH, num_trees)

    training_features, training_labels = get_features_labels(trainingSet)
    test_features, test_labels = get_features_labels(testSet)
    test_predictions, test_accuracy = get_inference_multiple_trees(baggedTrees, test_features, test_labels)
    train_predictions, train_accuracy = get_inference_multiple_trees(baggedTrees, training_features, training_labels)
    return train_accuracy, test_accuracy

def bagging(trainingSet, testSet):
    return run_bagging(trainingSet, testSet, 8, 30)

def run_randomForests(trainingSet, testSet, MAX_DEPTH, num_trees):
    rfTrees =create_bagged_trees(trainingSet, True, MAX_DEPTH, num_trees)

    training_features, training_labels = get_features_labels(trainingSet)
    test_features, test_labels = get_features_labels(testSet)
    test_predictions, test_accuracy=get_inference_multiple_trees(rfTrees, test_features, test_labels)
    train_predictions, train_accuracy=get_inference_multiple_trees(rfTrees, training_features, training_labels)
    return train_accuracy, test_accuracy

def randomForests(trainingSet, testSet):
    return run_randomForests(trainingSet, testSet, 8, 30)

def main():
    t0 = time.time()
    if len(sys.argv) != 4:
        print ("usage: python [filename] [training file name] [test file name] [model type 1(DT), 2(BT) or 3(RF)]")
    else:
        trainingSet = pd.read_csv(sys.argv[1])
        testSet= pd.read_csv(sys.argv[2])
        model_type = int(sys.argv[3])
        
        if model_type == 1:
            training_accuracy, testing_accuracy = decisionTree(trainingSet, testSet)
            print ("Training Accuracy DT:", training_accuracy)
            print ("Testing Accuracy DT:", testing_accuracy)
        elif model_type == 2:
            training_accuracy, testing_accuracy = bagging(trainingSet, testSet)
            print ("Training Accuracy BT:", training_accuracy)
            print ("Testing Accuracy BT:", testing_accuracy)
        elif model_type == 3:
            training_accuracy, testing_accuracy = randomForests(trainingSet, testSet)
            print ("Training Accuracy RF:", training_accuracy)
            print ("Testing Accuracy RF:", testing_accuracy)
        else:
            print ("incorrect model type", model_type)

    t1 = time.time()
    total = t1-t0
    # print ("total time taken for running code", total, "seconds")
            
if __name__ == '__main__':
    main()