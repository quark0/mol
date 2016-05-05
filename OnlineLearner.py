import numpy as np
import sys

class OnlineLearner:

    def __init__(self, num_feature, gamma, eta0=1., type='classification'):
        self.w = np.zeros(num_feature)
        self.gamma = gamma
        self.eta0 = eta0
        self.num_iter = 1

        if type == 'classification':
            self.update = self._update_classification
            self.predict = self._predict_classification
        if type == 'regression':
            self.update = self._update_regression
            self.predict = self._predict_regression

    def _update_regression(self, x, y_true):
        ''' 1/2*(y-<x,w>)^2 '''

        grad = x*(self.w.dot(x) - y_true)

        eta = self.eta0/(self.gamma*np.sqrt(self.num_iter)) # diminishing step size
        self.w = self.w - eta*grad
        self.num_iter += 1

    def _update_classification(self, x, y_true):
        ''' (1-y<x,w>)_+ '''

        if y_true*self.w.dot(x) < 1.:
            grad = self.gamma*self.w - y_true*x
        else:
            grad = self.gamma*self.w

        eta = self.eta0/(self.gamma*self.num_iter) # pegasos
        self.w = self.w - eta*grad
        self.num_iter += 1

    def _predict_classification(self, x):
        return +1 if self.w.dot(x) > 0 else -1

    def _predict_regression(self, x):
        return self.w.dot(x)

def evaluate_accuracy(Y_pred, Y_true):
    return np.mean(Y_pred == Y_true, axis=0)

def evaluate_squared_error(Y_pred, Y_true):
    return np.sqrt(np.mean((Y_pred-Y_true)**2, axis=0))

if __name__ == '__main__':

    from DataLoader import DataLoader

    try:
        if sys.argv[1] == 'mnist':
            data = DataLoader('mnist')
            training_size = 1000
            eta0 = 1
            gamma = 0.1
        if sys.argv[1] == 'cs':
            data = DataLoader('cs')
            training_size = 15
            eta0 = 0.05
            gamma = 5e-1
    except:
        print 'usage: python %s [mnist|cs]' % sys.argv[0]
        sys.exit(1)

    if data.type == 'classification':
        evaluate = evaluate_accuracy
    if data.type == 'regression':
        evaluate = evaluate_squared_error

    learners = [OnlineLearner(data.P, gamma, eta0=eta0, type=data.type) for _ in xrange(data.K)]

    Y_pred = np.zeros((training_size, data.K))
    Y_true = np.zeros((training_size, data.K))

    for i in xrange(training_size):
        x, y = data.next()
        for k, learner in enumerate(learners):
            Y_pred[i,k] = learner.predict(x)
            Y_true[i,k] = y[k]
            learner.update(x, y[k])

    print evaluate(Y_pred, Y_true)

