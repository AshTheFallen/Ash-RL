import os
from sklearn.preprocessing import StandardScaler
import numpy as np


def get_scaler(env):
    """
    create a scaler by playing a random episode and fitting the scaler on the observed states. can become more
    accurate by running the code for multiple episodes
    takes as an input:
    :param env: the environment you are working with - must have a step function with the standard state-reward-done-info return values
    :return:
    """

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



def maybe_make_dir(dir):
    """
    creating a directory if it does not exist - utility function
    :param dir: the address to the directory
    :return:
    """
    if not os.path.exists(dir):
        print('creating directory')
        os.mkdir(dir)
