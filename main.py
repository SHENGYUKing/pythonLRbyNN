# -*- coding:utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def generate_dataset(n_samples=1000, noise=0.1):
    """
    generate dataset
    :param n_samples: int, total number of generated examples
    :param noise: float, noise of dataset
    :return:
        X, np.array, features, size (number of examples, dim of features)
        y, np.array, labels, size (number of examples, 1)
    """
    X, y = make_moons(n_samples=n_samples, noise=0.1)
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    return X, y


def sigmoid(z):
    """
    compute the sigmoid function of z
    :param z: np.array
    :return:
        s, np.array, sigmoid of z
    """
    s = np.divide(1., 1. + np.exp(-1. * z))
    return s


def init_wb(dim):
    """
    initialize parameters of logistics regression
    :param dim: int, dim of feature
    :return:
        w, np.array, initialize with standard distribution
        b, float, initialize with 0
    """
    w = np.random.randn(dim).reshape(-1, 1)
    b = 0.
    assert w.shape == (dim, 1)
    assert b == 0 or 0.
    return w, b


def propagate(w, b, X, y):
    """
    implement the cost function and its gradient for the propagation, including forward and backward
    :param w: np.array, weights of features, size (dim of features, 1)
    :param b: float, bias of features
    :param X: np.array, features, size (dim of features, number of examples)
    :param y: np.array, true labels, size (1, number of examples)
    :return:
        cost, float, negative log-likelihood cost for logistic regression
        grad, dict, gradient of the loss
        grad.dw, np.array, gradient of the loss with respect to w, thus same shape as w
        grad.db, float, gradient of the loss with respect to b, thus same shape as b
    """
    m = X.shape[0]

    # forward propagate
    a = sigmoid(np.dot(w.T, X) + b)
    cost = -1.0 / X.shape[1] * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))

    # backward propagate
    dw = np.dot(X, (a - y).T) / X.shape[1]
    db = np.sum(a - y) / X.shape[1]
    assert dw.shape == w.shape
    assert db.dtype == float

    cost = np.squeeze(cost)
    grads = {"dw": dw,
             "db": db}

    return cost, grads


def optimize(w, b, X, y, num_iterations, learning_rate, show=False):
    """
    function to optimize w and b by gradient descent algorithm
    :param w: np.array, weights, size (dim of features, 1)
    :param b: float, bias
    :param X: np.array, features matrix, size (dim of features, number of examples)
    :param y: np.array, labels, size (1, number of examples)
    :param num_iterations: int, number of iterations of the optimization loop
    :param learning_rate: float, learning rate of the gradient descent update rule
    :param show: bool, True to print cost every 100 iter, default False
    :return:
        costs, list, cost every 100 iter
        params, dict, dictionary containing the weights w and bias b
        params.w, np.array, updated weights of the logistics regression function
        params,b, float, updated bias of the logistics regression function
        grads, dict, dictionary containing the gradients of the weights and bias with respect to the cost function
        grads.dw, np.array, the gradients of the weights with respect to the cost function
        grads,db, float, the gradients of the bias with respect to the cost function
    """
    costs = []

    for i in range(num_iterations):
        cost, grads = propagate(w, b, X, y)
        dw = grads["dw"]
        db = grads["db"]

        # update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            if show:
                print("cost after {} iter: {}".format(i, cost))

    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}

    return costs, params, grads


def predict(w, b, X):
    """
    predict the result of examples' features using learned logistic regression parameters (w, b)
    :param w: np.array, learned weights of LR function
    :param b: float, learned bias of LR function
    :param X: np.array, input examples' features, size (dim of features, number of examples)
    :return:
        y_pred, np.array, predictions for the input examples
    """
    m = X.shape[1]
    y_pred = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    a = sigmoid(np.dot(w.T, X) + b)
    a_mask = (a[0, :] > 0.5)
    a[0, :] = a[0, :] * a_mask
    assert y_pred.shape == (1, m)

    return y_pred


def model(X_train, y_train, X_test, y_test, num_iterations=1000, learning_rate=0.5, show=False):
    """
    build the logistic regression model by customized function
    :param X_train: np.array, examples for training, size (dim of features, num_train)
    :param y_train: np.array, true labels of training examples, size (1, num_train)
    :param X_test: np.array, examples for testing, size (dim of features, num_test)
    :param y_test: np.array, true labels of testing examples, size (1, num_test)
    :param num_iterations: int, number of iterations of the optimization loop, default 1000
    :param learning_rate: float, learning rate of the gradient descent update rule, default 0.5
    :param show: bool, True to print cost every 100 iter, default False
    :return:
        cache, dict, dictionary to save results of model
        cache.costs, list, list of cost every 100 iter
        cache.y_pred_train, np.array, prediction of training examples
        cache.y_pred_test, np.array, prediction of testing examples
        cache.w, np.array, learned weights of LR function
        cache.b, learned bias of LR function
        cache.num_iterations, number of iterations of the optimization loop
        cache.learning_rate, learning rate of the gradient descent update rule
    """
    w, b = init_wb(X_train.shape[0])

    costs, params, grads = optimize(w, b, X_train, y_train, num_iterations, learning_rate, show)
    w = params["w"]
    b = params["b"]

    y_pred_train = predict(w, b, X_train)
    y_pred_test = predict(w, b, X_test)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_pred_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_pred_test - y_test)) * 100))

    cache = {
        "costs": costs,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "w": w,
        "b": b,
        "num_iterations": num_iterations,
        "learning_rate": learning_rate
    }
    return cache


def plot_cost(costs, learning_rate):
    costs = np.squeeze(costs)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def main():
    np.random.seed(2021)
    X, y = generate_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    X_train, X_test = X_train.T, X_test.T
    y_train, y_test = y_train.T, y_test.T
    cache = model(X_train, y_train, X_test, y_test, num_iterations=1000, learning_rate=0.5, show=True)
    plot_cost(cache["costs"], cache["learning_rate"])


if __name__ == '__main__':
    main()
