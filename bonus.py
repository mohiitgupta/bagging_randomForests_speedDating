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
        if count != 0:
            prob = count/total_count
            sum_prob += pow(prob,2)
    return 1 - sum_prob

def calculate_gini(labels, weights):
    total_count = 0
    count = [0 for i in range(LABELS_LEN)]
    for i in range(len(labels)):
        label = labels.iloc[i]
        count[label] += weights.iloc[i]
        total_count += weights.iloc[i]
    return find_gini(count, total_count)

def get_features_labels(dataset):
    features = dataset.drop(columns = ['decision'])
    labels = dataset['decision']
    return features, labels

def predict_label(labels, weights):
    # print ("labels shape", labels.shape)
    # print ("weights shape ", weights.shape)
    count = [0 for i in range(LABELS_LEN)]
    max_label=0
    max_count=0
    total=0
    for i in range(len(labels)):
        label = labels.iloc[i]
        # print ("weight at ",i, "is", weights[i])
        # print ("count",count[label])
        count[label] += weights.iloc[i]
        total += weights.iloc[i]
        if count[label] > max_count:
            max_count = count[label]
            max_label = label

    return max_label, 100*max_count/(1.0*total)

def count_branch(attr, attr_value, dataframe):
    dataframe_f = dataframe[dataframe[attr]==attr_value]
    count_array = []
    for label in range(LABELS_LEN):
        count_array.append(dataframe_f[dataframe_f['decision']==label]['weights'].sum())
    
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

def create_decision_tree(trainingSet, depth, is_random_forest, MAX_DEPTH):   
    weights = trainingSet['weights'] 
    predicted_label, confidence = predict_label(trainingSet['decision'], weights)
    if (depth >= MAX_DEPTH) or (len(trainingSet) < 50) or (confidence == 100):
        return Node(None, predicted_label)

    # print ("\ndepth", depth, "length training set", len(trainingSet['decision']))
    gini_sample = calculate_gini(trainingSet['decision'], weights)
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
        if attr != 'decision' and attr != 'weights':
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
        node.left = create_decision_tree(left_trainingSet, depth+1, is_random_forest, MAX_DEPTH)
    else:
        node.left = Node(None, predicted_label)
    
    if len(right_trainingSet) > 0:
        node.right = create_decision_tree(right_trainingSet, depth+1, is_random_forest, MAX_DEPTH)
    else:
        node.right = Node(None, predicted_label)
    return node


def get_error_hypothesis(root, trainingSet):
    error = 0
    predicted_labels = []
    weights = trainingSet['weights']
    labels = trainingSet['decision']
    sum_weights = sum(weights)
    for i in range(len(trainingSet)):
        predicted_label = get_label_decision_tree(root, trainingSet.iloc[i])
        predicted_labels.append(predicted_label)
        true_label = labels[i]
        if true_label != predicted_label:
            error += weights[i]
    error = error * 1.0 / sum_weights
    return error, predicted_labels

def get_inference_boosted_trees(hypothesis, test_points, test_labels, alphas):
    correct_points = 0
    predictions = []

    for i in range(len(test_points)):
        output = [0 for j in range(LABELS_LEN)]
        predicted_label = 0
        max_value = -1000
        for j,root in enumerate(hypothesis):
            label = get_label_decision_tree(root, test_points.iloc[i])
            output[label] += alphas[j]
            if output[label] > max_value:
                max_value = output[label]
                predicted_label = label
        
        true_label = test_labels[i]
        if true_label == predicted_label:
            correct_points += 1
        predictions.append(predicted_label)

    accuracy = correct_points/len(test_points)
    return predictions, accuracy

def create_boosted_trees(trainingSet, is_random_forest, MAX_DEPTH, num_trees):
    hypothesis = []
    alphas = []
    weights = [1.0/len(trainingSet)]*len(trainingSet)
    for i in range(num_trees):
        trainingSet['weights'] = weights
        root = create_decision_tree(trainingSet, 0, is_random_forest, MAX_DEPTH)
        hypothesis.append(root)
        error, predicted_labels = get_error_hypothesis(root, trainingSet)
        alpha = math.log((1-error)*1.0/error)
        alphas.append(alpha)
        for j, label in enumerate(trainingSet['decision'].values):
            if label != predicted_labels[j]:
                weights[j] = weights[j] * math.exp(alpha)
        sum_weights = sum(weights)
        # print ("original sum weights ", sum_weights)
        weights[:] = [weight*1.0/sum_weights for weight in weights]
    return hypothesis, alphas


def boosted_decision_trees(trainingSet, testSet, MAX_DEPTH, num_trees):
    is_random_forest = False
    boostedTrees, alphas = create_boosted_trees(trainingSet, is_random_forest, MAX_DEPTH, num_trees)

    training_features, training_labels = get_features_labels(trainingSet)
    test_features, test_labels = get_features_labels(testSet)
    test_predictions, test_accuracy = get_inference_boosted_trees(boostedTrees, test_features, test_labels, alphas)
    train_predictions, train_accuracy = get_inference_boosted_trees(boostedTrees, training_features, training_labels, alphas)
    return train_accuracy, test_accuracy

def main():
    t0 = time.time()
    if len(sys.argv) != 4:
        print ("usage: python [filename] [training file name] [test file name] [model type 1(Boosted Decision Trees)]")
    else:
        trainingSet = pd.read_csv(sys.argv[1])
        testSet= pd.read_csv(sys.argv[2])
        model_type = int(sys.argv[3])
        
        if model_type == 1:
            training_accuracy, testing_accuracy = boosted_decision_trees(trainingSet, testSet, 3, 30)
            print ("Training Accuracy Boosted Decision Tree:", round(training_accuracy,2))
            print ("Testing Accuracy Boosted Decision Tree:", round(testing_accuracy,2))
        else:
            print ("incorrect model type", model_type)

    t1 = time.time()
    total = t1-t0
    print ("total time taken for running code", total, "seconds")
            
if __name__ == '__main__':
    main()