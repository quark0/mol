import cPickle as pickle
import gzip
import numpy as np
from scipy import io

class DataLoader:

    def __init__(self, dataset):

        if dataset == 'mnist':
            with gzip.open('mnist.pkl.gz', 'rb') as f:
                train_set, _, _ = pickle.load(f)

            self.X, y = train_set[0], train_set[1]
            self.N, self.P = self.X.shape
            self.K = np.unique(y).shape[0]

            # Y is an N-by-K matrix of {-1,+1}
            self.Y = -np.ones((self.N, self.K))
            for i, label in enumerate(y):
                self.Y[i,label] = 1

            self.type = 'classification'

        elif dataset == 'cs':
            train_set = io.loadmat('computer_survey_190.mat')
            self.Y = train_set['Y']
            self.X = train_set['X']
            self.N, self.P = self.X.shape
            self.K = self.Y.shape[1]

            self.type = 'regression'

        else:
            raise NameError('invalid dataset name')

        self.cur = 0

    def __iter__(self):
        return self

    def next(self):
        if self.cur > len(self.X):
            raise StopIteration
        else:
            self.cur += 1
            return self.X[self.cur], self.Y[self.cur]

