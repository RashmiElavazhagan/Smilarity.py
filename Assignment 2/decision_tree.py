# -*- coding: utf-8 -*-
"""decision_tree.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1K6Cl1-HqvaBHj7roZ-ghOPCmlvSTGlyO
"""

# -------------------------------------------------------------------------
# AUTHOR: Rashmi E.
# FILENAME: decision_tree.py
# SPECIFICATION: A program that trains the model and tests to compute the accuracies and output the average accuracy for the iterations.
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 1.25 - 1.5 hours
# -----------------------------------------------------------*/

# Importing necessary Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Dataset file paths
data_files = ['/content/cheat_training_1.csv', '/content/cheat_training_2.csv']

# Iterating over each dataset for training and testing
for data_file in data_files:

    X = []
    Y = []

    # Reading the dataset without headers using Pandas
    df = pd.read_csv(data_file, sep=',', header=0)
    data_training = np.array(df.values)[:, 1:]  # Creating a training matrix omitting the ID (NumPy)

    # Transforming original training features to numbers
    for record in data_training:
        record_array = [1 if record[0] == 'Yes' else 0, 1 if record[1] == 'Single' else 0,
                        float(str(record[2]).replace('k', ''))]
        X.append(record_array)

    # Transforming original training classes to numbers
    for record in data_training:
        Y.append(1 if record[3] == 'Yes' else 2)

    accuracies = []

    # Running training and testing tasks 10 times
    for i in range(10):

        # Fitting the decision tree model using Gini index and no max_depth constraint
        clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
        clf = clf.fit(X, Y)

        # Plotting the decision tree
        tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'],
                       class_names=['Yes', 'No'], filled=True, rounded=True)
        plt.show()

        # Reading the test data
        df = pd.read_csv('cheat_test.csv', sep=',', header=0)
        data_test = np.array(df.values)[:, 1:]

        accuracy_counter = 0

        # Evaluating model accuracy on test data
        for data in data_test:
            test_record = [1 if data[0] == 'Yes' else 0, 1 if data[1] == 'Single' else 0,
                           float(str(data[2]).replace('k', ''))]
            class_predicted = clf.predict([test_record])[0]

            # Comparing predicted label with true label
            if class_predicted == 1 and data[3] == 'Yes':
                accuracy_counter += 1

        accuracies.append(accuracy_counter / len(data_test))

    # Outputting average accuracy for the model
    print("Average accuracy for model trained on", data_file, "is", sum(accuracies) / len(accuracies))