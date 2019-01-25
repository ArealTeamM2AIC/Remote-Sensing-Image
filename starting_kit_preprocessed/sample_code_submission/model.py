import numpy as np
import pickle
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from os.path import isfile

def requires_grad(p):
    return p.requires_grad


class baselineModel(BaseEstimator):
    def __init__(self, max_depth=None):
        """
        Using DecisionTreeClassifier from sklearn as Baseline Model
        Has one parameter which is the max depth of the tree (base value of 5)
        """
        super(baselineModel, self).__init__()
        self.classifier = DecisionTreeClassifier(max_depth=max_depth)
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False

    def fit(self, X, y):
        self.num_train_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1]
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = y.shape[0]
        if y.ndim>1: self.num_labels = y.shape[1]
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        self.is_trained=True
        self.classifier.fit(X, y)

    def predict(self, X):
        num_test_samples = X.shape[0]
        if X.ndim>1:
            num_feat = X.shape[1]
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        y = np.zeros([num_test_samples, self.num_labels])
        return self.classifier.predict(X)

    def save(self, path="./"):
        pickle.dump(self, open(path + '_model.pickle', "wb"))

    def load(self, path="./"):
        modelfile = path + '_model.pickle'
        if isfile(modelfile):
            with open(modelfile, 'rb') as f:
                self = pickle.load(f)
            print("Model reloaded from: " + modelfile)
        return self
