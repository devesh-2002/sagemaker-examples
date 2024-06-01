from __future__ import print_function

import argparse
import joblib
import os
import pandas as pd

from sklearn import tree


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.
    parser.add_argument('--max_leaf_nodes', type=int, default=-1)
    parser.add_argument('--n_estimators', type=int, default=100)
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str,default='output')
    parser.add_argument('--model-dir', type=str,default='model')
    parser.add_argument('--train', type=str)

    args = parser.parse_args()

#     input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
#     if len(input_files) == 0:
#         raise ValueError(('There are no files in {}.\n' +
#                           'This usually indicates that the channel ({}) was incorrectly specified,\n' +
#                           'the data specification in S3 was incorrectly specified or the role specified\n' +
#                           'does not have permission to access the data.').format(args.train, "train"))
#     raw_data = [ pd.read_csv(file, header=None, engine="python") for file in input_files ]
#     train_data = pd.concat(raw_data)

#     train_y = train_data.iloc[:, 0]
#     train_X = train_data.iloc[:, 1:]
    train_X = pd.read_csv('X_train.csv')
    train_y = pd.read_csv('y_train.csv')
    max_leaf_nodes = args.max_leaf_nodes

    # Now use scikit-learn's decision tree classifier to train the model.
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes)
    clf = clf.fit(train_X, train_y)

    joblib.dump(clf, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf
