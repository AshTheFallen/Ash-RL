import numpy as np
import os
from sklearn.preprocessing import StandardScaler


# create a scaler by playing a random episode and fitting the scaler on the observed states.
# can become more accurate by running the code for multiple episodes
def get_scaler(env):
    states = []
    for i in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


class DQNAgent:
    class LinearModel:
        def __init__(self, input_dim, n_actions):
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

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.9
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = LinearModel(state_size, action_size)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
