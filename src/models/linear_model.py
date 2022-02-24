import numpy as np


class LinearModel:
    """
    A simple linear model that uses stochastic gradient descent with momentum for value approximation
    """

    def __init__(self, input_dim, n_actions):
        """
        constructor for the model
        :param input_dim: the input dimensions of the state vector of the environment
        :param n_actions: the number of actions possible in the environment
        """
        self.W = np.random.randn(input_dim, n_actions) / np.sqrt(input_dim)
        self.b = np.zeros(n_actions)
        # used for momentum in gradient descent
        self.vW = 0
        self.vb = 0
        self.losses = []

    def predict(self, X):
        # X must be of shape N * D and 2D
        assert (len(X.shape) == 2)
        return X.dot(self.W) + self.b

    def sgd(self, X, Y, learning_rate=0.001, momentum=0.9):
        # X must be of shape N * D and 2D
        assert (len(X.shape) == 2)
        num_values = np.prod(Y.shape)
        Y = np.float128(Y)
        Yhat = np.float128(self.predict(X))
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values

        # updating the momentum
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        # update params
        self.W += self.vW
        self.b += self.vb

        mse = np.float128(np.mean((Yhat - Y) ** 2))
        self.losses.append(mse)

    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)
