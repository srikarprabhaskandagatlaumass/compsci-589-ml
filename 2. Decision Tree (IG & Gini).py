# Srikar Prabhas Kandagatla (34964700)
"""
This code implements the Decision Tree algorithm in Python. Executes less than
30 sec most of the time.

I have executed the following code in Google Colab with CPU runtime type. Please feel
free to use this code. It can be executed in any IDE that has the required packages installed
(Numpy, Pandas, Matplotlib, and scikit-learn).

This code was completed and submitted for the course COMPSCI 589: Machine Learning as part of
Homework Assignment 1, University of Massachusetts, Amherst.
"""

# Importing the Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

# Class representing a node in the decision tree
class DecisionNode:
    def __init__(self, feature=None, descendants=None, category=None):
        self.feature = feature
        self.descendants = descendants or {}
        self.category = category
"""
The DecisionNode class represents a node in the decision tree. It stores the 
feature used for splitting, its descendants (child nodes), and the category 
(for leaf nodes). This structure allows the tree to be traversed and decisions 
to be made based on the features of the input data.
"""

# Calculates the entropy of a sample (dataset)
def entropy(sample):
    label_counts = Counter(row[-1] for row in sample)
    total_samples = len(sample)
    return -sum((count / total_samples) * np.log2(count / total_samples) 
                for count in label_counts.values() if count > 0)
"""
The entropy function calculates the entropy of a dataset, which measures the 
impurity or uncertainty. Higher entropy indicates a more mixed set of classes,
while lower entropy indicates a more homogeneous set. This measure is crucial 
for determining the best feature to split on in the decision tree.
"""

# Calculates the Information Gain to select the best feature
def information_gain(sample, available_features, all_features):
    def feature_entropy(feature):
        feature_idx = all_features.index(feature)
        feature_values = set(row[feature_idx] for row in sample)
        
        weighted_entropy = sum(
            len(subset) / len(sample) * entropy(subset)
            for value in feature_values
            if (subset := [row for row in sample if row[feature_idx] == value])
        )
        return entropy(sample) - weighted_entropy
    return max(available_features, key=feature_entropy)
"""
The information_gain function calculates the information gain for each available
feature. It chooses the feature that provides the highest information gain, 
which is the greatest reduction in entropy. This method helps in selecting 
the most informative feature for splitting at each node of the decision tree.
"""

# Similar to Information Gain, Gini Criterion selects the best attribute
def gini_criterion(sample, available_features, all_features):
    def feature_gini(feature):
        feature_idx = all_features.index(feature)
        feature_values = set(row[feature_idx] for row in sample)
        
        weighted_gini = sum(
            len(subset) / len(sample) * (1 - sum((Counter(row[-1] for row in subset)[label] / len(subset))**2 for label in set(row[-1] for row in subset)))
            for value in feature_values
            if (subset := [row for row in sample if row[feature_idx] == value])
        )
        
        return 1 - sum((Counter(row[-1] for row in sample)[label] / len(sample))**2 for label in set(row[-1] for row in sample)) - weighted_gini
    return max(available_features, key=feature_gini)

# Chooses between information gain and Gini criterion as the splitting criterion
def split_criterion(sample, available_features, all_features, method):
    criterions = {
        0: information_gain,
        1: gini_criterion
    }
    return criterions[method](sample, available_features, all_features)

# Recursively constructs the decision tree
def construct_tree(sample, available_features, all_features, method=0):
    labels = [row[-1] for row in sample]
    
    if len(set(labels)) == 1:
        return DecisionNode(category=labels[0])
    
    if not available_features:
        return DecisionNode(category=Counter(labels).most_common(1)[0][0])
    
    best_feature = split_criterion(sample, available_features, all_features, method)
    root = DecisionNode(feature=best_feature)
    
    feature_idx = all_features.index(best_feature)
    for value in set(row[feature_idx] for row in sample):
        subset = [row for row in sample if row[feature_idx] == value]
        if subset:
            remaining_features = [f for f in available_features if f != best_feature]
            root.descendants[value] = construct_tree(subset, remaining_features, all_features, method)
        else:
            root.descendants[value] = DecisionNode(category=Counter(labels).most_common(1)[0][0])
    
    return root

# Classifies a single sample using the decision tree
def classify_sample(tree, sample, features):
    if tree.category is not None:
        return tree.category
    
    feature_value = sample[features.index(tree.feature)]
    
    if feature_value not in tree.descendants:
        descendant_categories = [d.category for d in tree.descendants.values() if d.category]
        return Counter(descendant_categories).most_common(1)[0][0] if descendant_categories else \
               classify_sample(next(iter(tree.descendants.values())), sample, features)
    
    return classify_sample(tree.descendants[feature_value], sample, features)

# Calculates the accuracy of the decision tree on a dataset
def assess_accuracy(tree, dataset, features):
    correct_count = sum(1 for row in dataset if classify_sample(tree, row, features) == row[-1])
    return correct_count / float(len(dataset))

# Builds a decision tree and evaluates its performance
def decision_tree(X_train, X_test, y_train, y_test, header, method):
    features = list(header[:-1])
    available_features = features[:]

    train_set = np.column_stack((X_train, y_train))
    test_set = np.column_stack((X_test, y_test))

    tree = construct_tree(train_set, available_features, features, method)

    train_accuracy = assess_accuracy(tree, train_set, features)
    test_accuracy = assess_accuracy(tree, test_set, features)
    return train_accuracy, test_accuracy

#----Main execution----
# Reading the dataset using read_csv
dataset = pd.read_csv('/content/car.csv', header=None)
header = dataset.iloc[0].tolist()
full_data = dataset.iloc[1:].values

X = full_data[:, :-1]
y = full_data[:, -1]

ig_train_accuracies = []
ig_test_accuracies = []
gini_train_accuracies = []
gini_test_accuracies = []

for _ in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    
    ig_train_acc, ig_test_acc = decision_tree(X_train, X_test, y_train, y_test, header, method=0)
    gini_train_acc, gini_test_acc = decision_tree(X_train, X_test, y_train, y_test, header, method=1)
    
    ig_train_accuracies.append(ig_train_acc)
    ig_test_accuracies.append(ig_test_acc)
    gini_train_accuracies.append(gini_train_acc)
    gini_test_accuracies.append(gini_test_acc)
"""
This code splits data into features and a target variable, then iterates 100
times, building and evaluating decision trees using Information Gain and
Gini Criterion. It stores training and testing accuracies for each method,
allowing comparison across multiple random data splits.
"""

# Plotting results
# Plotting results with mean and standard deviation annotations
plt.figure(figsize=(10, 6))
ig_train_mean = np.mean(ig_train_accuracies)
ig_train_std = np.std(ig_train_accuracies)
plt.hist(ig_train_accuracies, bins=20, edgecolor='black')
plt.axvline(ig_train_mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {ig_train_mean:.4f}')
plt.axvline(ig_train_mean + ig_train_std, color='green', linestyle='dotted', linewidth=1, label=f'Mean + Std: {ig_train_mean + ig_train_std:.4f}')
plt.axvline(ig_train_mean - ig_train_std, color='green', linestyle='dotted', linewidth=1, label=f'Mean - Std: {ig_train_mean - ig_train_std:.4f}')
plt.xlabel('Accuracy')
plt.ylabel('Accuracy Frequency on Training Data')
plt.title(f'Information Gain - Training Accuracy Distribution\nMean: {ig_train_mean:.4f}, Std: {ig_train_std:.4f}')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
ig_test_mean = np.mean(ig_test_accuracies)
ig_test_std = np.std(ig_test_accuracies)
plt.hist(ig_test_accuracies, bins=20, edgecolor='black')
plt.axvline(ig_test_mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {ig_test_mean:.4f}')
plt.axvline(ig_test_mean + ig_test_std, color='green', linestyle='dotted', linewidth=1, label=f'Mean + Std: {ig_test_mean + ig_test_std:.4f}')
plt.axvline(ig_test_mean - ig_test_std, color='green', linestyle='dotted', linewidth=1, label=f'Mean - Std: {ig_test_mean - ig_test_std:.4f}')
plt.xlabel('Accuracy')
plt.ylabel('Accuracy Frequency on Testing Data')
plt.title(f'Information Gain - Testing Accuracy Distribution\nMean: {ig_test_mean:.4f}, Std: {ig_test_std:.4f}')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
gini_train_mean = np.mean(gini_train_accuracies)
gini_train_std = np.std(gini_train_accuracies)
plt.hist(gini_train_accuracies, bins=20, edgecolor='black')
plt.axvline(gini_train_mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {gini_train_mean:.4f}')
plt.axvline(gini_train_mean + gini_train_std, color='green', linestyle='dotted', linewidth=1, label=f'Mean + Std: {gini_train_mean + gini_train_std:.4f}')
plt.axvline(gini_train_mean - gini_train_std, color='green', linestyle='dotted', linewidth=1, label=f'Mean - Std: {gini_train_mean - gini_train_std:.4f}')
plt.xlabel('Accuracy')
plt.ylabel('Accuracy Frequency on Training Data')
plt.title(f'Gini Criterion - Training Accuracy Distribution\nMean: {gini_train_mean:.4f}, Std: {gini_train_std:.4f}')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
gini_test_mean = np.mean(gini_test_accuracies)
gini_test_std = np.std(gini_test_accuracies)
plt.hist(gini_test_accuracies, bins=20, edgecolor='black')
plt.axvline(gini_test_mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {gini_test_mean:.4f}')
plt.axvline(gini_test_mean + gini_test_std, color='green', linestyle='dotted', linewidth=1, label=f'Mean + Std: {gini_test_mean + gini_test_std:.4f}')
plt.axvline(gini_test_mean - gini_test_std, color='green', linestyle='dotted', linewidth=1, label=f'Mean - Std: {gini_test_mean - gini_test_std:.4f}')
plt.xlabel('Accuracy')
plt.ylabel('Accuracy Frequency on Testing Data')
plt.title(f'Gini Criterion - Testing Accuracy Distribution\nMean: {gini_test_mean:.4f}, Std: {gini_test_std:.4f}')
plt.legend()
plt.show()

#Printing the Accuracies
print(f"Avg Information Gain - Train: {np.mean(ig_train_accuracies):.4f}, Test: {np.mean(ig_test_accuracies):.4f}")
print(f"Avg Gini Criterion - Train: {np.mean(gini_train_accuracies):.4f}, Test: {np.mean(gini_test_accuracies):.4f}")