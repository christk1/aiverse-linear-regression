import numpy as np

def COD(x, y, theta):
    mean_of_ys = np.sum(y) / len(y)
    y_variation = np.sum(np.square(y - mean_of_ys))

    test_hypothesis = np.dot(x, theta)
    sq_error = np.sum(np.square(test_hypothesis - y))
    r2 = 1 - (sq_error / y_variation)
    return r2

def adj_r2(num_of_params, num_of_feats, r2):
    adj_r2 = 1 - (1 - r2) * (num_of_feats - 1) / \
        (num_of_feats - num_of_params - 1)
    return adj_r2

def ZNorm(x):
    mi = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_norm = (x - mi) / sigma
    return x_norm, mi, sigma


def computeCost(y, hypothesis):
    train_len = len(y)
    J = 1 / (2 * train_len) * np.sum(np.square(hypothesis - y))
    return J


def gradientDescent(x_train, x_test, y_train, y_test, theta, alpha, num_iters, alpha_increase):
    train_len = len(y_train)
    test_lenn = len(y_test)
    history_train = np.zeros(num_iters)
    history_test = np.zeros(num_iters)

    for i in range(num_iters):
        # train
        hypothesis = np.dot(x_train, theta)
        history_train[i] = computeCost(y_train, hypothesis)
        theta -= alpha * (1 / train_len) * np.dot(x_train.T, (hypothesis - y_train))
        # test
        hypothesis = np.dot(x_test, theta)
        history_test[i] = computeCost(y_test, hypothesis)

        if i % alpha_increase == 0 and i != 0:
            alpha *= .1
            alpha_increase = i * 10

    return theta, history_train, history_test
