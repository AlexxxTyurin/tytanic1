import math
import numpy as np
import pandas as pd
import sys
import os


def add_ones(array):
    new_array = np.ones([array.shape[0], array.shape[1]+1])
    new_array[:, 1:] = array
    return new_array


# Counts the gradient of function
def gradient(x, y, theta):
    grad = sum((sigmoid(x, theta) - y) * x)
    return np.array([[el] for el in grad]) / len(y[:, 0])


# Performs the gradient descent
def gradient_descent(x, y, theta, alpha, num_iter):
    ls = np.zeros(num_iter)
    for i in range(num_iter):
        ls[i] = logistic_cost(x, y, theta)

        theta -= gradient(x, y, theta) * alpha / len(y)

    return theta, ls


# Counts the cost function
def logistic_cost(hypothesis, real_results):
    return -sum(sum(real_results * np.log(hypothesis) + (1 - real_results) * np.log(1 - hypothesis))) / \
           hypothesis.shape[0]


def max_index(mass):
    maximum = 0
    index = 0
    for i in range(len(mass)):
        if mass[i] > maximum:
            maximum = mass[i]
            index = i
    return index


def mean(array):
    try:
        array = np.array(array)
        return sum(sum(array)) / (array.shape[0] * array.shape[1])
    except TypeError:
        return sum(array) / len(array)


def mistakes(hypothesis, real_results):
    n = 0
    for i in range(hypothesis.shape[0]):
        if max_index(hypothesis[i, :]) == real_results[i]:
            n += 1
    return n / hypothesis.shape[0]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Counts the cost function with regularization
def reg_logistic_cost(hypothesis, real_results, weights_container):
    weights_cost = sum([sum(sum(el)) for el in weights_container]) + hypothesis.shape[0]
    return -sum(sum(real_results * np.log(hypothesis) + (1 - real_results) * np.log(1 - hypothesis))) / hypothesis.shape[0] + weights_cost


def element_svm_negative_cost(hypothesis):
    if hypothesis <= -1:
        return 0
    else:
        return 3 + 3 * hypothesis


def variance(array):
    try:
        array = np.array(array)
        m = mean(array)
        return sum(sum((array - m)**2)) / (array.shape[0] * array.shape[1])
    except TypeError:
        return sum((array - m) ** 2) / len(array)


def generate_weights(numbers):
    weights = []
    for el in numbers:
        weights.append(np.random.sample(el))
    return weights


class NN:
    def __init__(self, train_data, test_data, train_results, test_results,
                 weights):
        self.train_data = (train_data - mean(train_data)) / (variance(train_data)**0.5)
        self.test_data = (test_data - mean(test_data)) / (variance(test_data)**0.5)
        self.train_results = np.array([np.eye(10)[el[0], :] for el in train_results])
        self.real_test_results = test_results
        self.test_results = np.array([np.eye(10)[el[0], :] for el in test_results])
        self.weights = weights
        self.layers = []
        self.activations = []
        self.delta_weights = []
        self.layer_errors = []

    def forward_propagation(self, data):
        self.activations.clear()
        self.layers.clear()
        self.activations.append(data)
        # print(self.activations[0].shape)
        for i in range(1, len(self.weights)+1):
            self.layers.append(np.matmul(self.activations[i-1], self.weights[i-1]))
            if i == len(self.weights):
                self.activations.append(sigmoid(self.layers[i-1]))
                # print(self.activations[i].shape)
            else:
                self.activations.append(sigmoid(add_ones(self.layers[i-1])))
                # print(self.activations[i].shape)

    def submit(self, data):
        self.forward_propagation(data)
        submission = pd.Series([max_index(self.activations[-1][i, :]) for i in range(self.activations[-1].shape[0])])
        submission = pd.DataFrame({"ImageId":np.arange(1, 28001), "Label":submission})
        submission.to_csv('/Users/vladikturin/Desktop/pycharm_projects/recognition/submit.csv', index=False)


    def mini_batch_back_propagation(self, epochs, alpha, lmbda, batch_size):
        # This is a "mini-batch" gradient descent algorithm. This means that on an iteration we calculate the error on
        # batches and after that we adjust the weights according to the error of this batch

        # We collect the information about the errors in order to see how the model is improving
        errors = np.zeros([epochs])
        for j in range(epochs):
            self.forward_propagation(self.test_data)

            sys.stdout.write(f"Epoch: {j}, error: {logistic_cost(self.activations[-1], self.test_results)}\n")
            # sys.stdout.write(self.hypothesis[0, :])
            sys.stdout.flush()

            # We propagate forward and then calculate the error on validation data
            # self.forward_propagation(self.validation_data)
            errors[j] = logistic_cost(self.activations[-1], self.test_results)

            # We collect the info about delta_weights
            self.delta_weights = [np.zeros(el.shape) for el in self.weights]

            self.forward_propagation(self.train_data)

            for i in range(0, int(self.train_data.shape[0]), batch_size):

                # On every iteration we calculate the errors in layers, that is why on every iteration we should
                # clear the information about the errors and set it to zero
                self.layer_errors = [0 for el in self.weights]

                # The output error(which is the last error in self.layer_errors) is calculated as the difference between
                # the hypothesis(self.activations[-1]) and the rights results(self.train_results)
                self.layer_errors[-1] = self.activations[-1][i:i+batch_size, :] - self.train_results[i:i+batch_size, :]

                # Here we compute the layer errors. It means that we have to change the result in a layer by a
                # particular amount
                for t in range(len(self.activations)-3, -1, -1):
                    if t == len(self.activations)-3:
                        _ = np.matmul(self.layer_errors[t+1], np.transpose(self.weights[t+1]))
                        self.layer_errors[t] = _ * sigmoid_derivative(add_ones(self.layers[t][i:i+batch_size]))

                    else:
                        _ = np.matmul(self.layer_errors[t+1][:, 1:], np.transpose(self.weights[t+1]))
                        self.layer_errors[t] = _ * sigmoid_derivative(add_ones(self.layers[t][i:i+batch_size]))

                # Here we compute how we should change the weights
                for t in range(len(self.delta_weights)):
                    if t == len(self.delta_weights) - 1:
                        h = np.transpose(np.matmul(np.transpose(self.layer_errors[t]), self.activations[t][i:i + batch_size, :]))
                        self.delta_weights[t] += h + self.weights[t] / self.train_data.shape[0]
                    else:
                        h = np.transpose(np.matmul(np.transpose(self.layer_errors[t][:, 1:]), self.activations[t][i:i + batch_size, :]))
                        self.delta_weights[t] += h + lmbda * self.weights[t] / self.train_data.shape[0]

            # Here we change the weights by alpha * delta_weights
            for t in range(len(self.delta_weights)):
                self.weights[t] -= alpha * self.delta_weights[t] / (self.train_data.shape[0] / batch_size)

        return errors

    def classical_back_propagation(self, epochs, alpha, lmbda):
        # This is a "mini-batch" gradient descent algorithm. This means that on an iteration we calculate the error on
        # batches and after that we adjust the weights according to the error of this batch

        # We collect the information about the errors in order to see how the model is improving
        errors = np.zeros([epochs])
        for j in range(epochs):
            self.forward_propagation(self.test_data)

            sys.stdout.write(f"Epoch: {j}, error: {logistic_cost(self.activations[-1], self.test_results)}\n")
            # sys.stdout.write(self.hypothesis[0, :])
            sys.stdout.flush()

            # We propagate forward and then calculate the error on validation data
            errors[j] = logistic_cost(self.activations[-1], self.test_results)

            # We collect the info about delta_weights
            self.delta_weights = [np.zeros(el.shape) for el in self.weights]

            self.forward_propagation(self.train_data)

            # On every iteration we calculate the errors in layers, that is why on every iteration we should
            # clear the information about the errors and set it to zero
            self.layer_errors = [0 for el in self.weights]

            # The output error(which is the last error in self.layer_errors) is calculated as the difference between
            # the hypothesis(self.activations[-1]) and the rights results(self.train_results)
            self.layer_errors[-1] = self.activations[-1] - self.train_results

            # Here we compute the layer errors. It means that we have to change the result in a layer by a
            # particular amount
            for t in range(len(self.activations) - 3, -1, -1):
                if t == len(self.activations) - 3:
                    _ = np.matmul(self.layer_errors[t + 1], np.transpose(self.weights[t + 1]))
                    self.layer_errors[t] = _ * sigmoid_derivative(add_ones(self.layers[t]))

                else:
                    _ = np.matmul(self.layer_errors[t + 1][:, 1:], np.transpose(self.weights[t + 1]))
                    self.layer_errors[t] = _ * sigmoid_derivative(add_ones(self.layers[t]))

            # Here we compute how we should change the weights
            for t in range(len(self.delta_weights)):
                if t == len(self.delta_weights) - 1:
                    h = np.transpose(np.matmul(np.transpose(self.layer_errors[t]), self.activations[t]))
                    self.delta_weights[t] += h + lmbda * self.weights[t] / self.train_data.shape[0]
                else:
                    h = np.transpose(
                        np.matmul(np.transpose(self.layer_errors[t][:, 1:]), self.activations[t]))
                    self.delta_weights[t] += h + lmbda * self.weights[t] / self.train_data.shape[0]

            # Here we change the weights by alpha * delta_weights
            for t in range(len(self.delta_weights)):
                self.weights[t] -= alpha * self.delta_weights[t]

        self.forward_propagation(self.test_data)
        print(mistakes(self.activations[-1], self.real_test_results))
        return errors

    def get_result(self, alpha, lmbda, batch_size, n):
        j = 0
        self.forward_propagation(self.test_data)
        sys.stdout.write(f"Epoch: {j}, error: {logistic_cost(self.activations[-1], self.test_results)}\n")
        # sys.stdout.write(self.hypothesis[0, :])
        sys.stdout.flush()

        while logistic_cost(self.activations[-1], self.test_results) > n or math.isnan(logistic_cost(self.activations[-1], self.test_results)):

            # We collect the info about delta_weights
            self.delta_weights = [np.zeros(el.shape) for el in self.weights]

            self.forward_propagation(self.train_data)

            for i in range(0, int(self.train_data.shape[0]), batch_size):

                # On every iteration we calculate the errors in layers, that is why on every iteration we should
                # clear the information about the errors and set it to zero
                self.layer_errors = [0 for el in self.weights]

                # The output error(which is the last error in self.layer_errors) is calculated as the difference between
                # the hypothesis(self.activations[-1]) and the rights results(self.train_results)
                self.layer_errors[-1] = self.activations[-1][i:i+batch_size, :] - self.train_results[i:i+batch_size, :]

                # Here we compute the layer errors. It means that we have to change the result in a layer by a
                # particular amount
                for t in range(len(self.activations) - 3, -1, -1):
                    if t == len(self.activations) - 3:
                        _ = np.matmul(self.layer_errors[t + 1], np.transpose(self.weights[t + 1]))
                        self.layer_errors[t] = _ * sigmoid_derivative(add_ones(self.layers[t][i:i + batch_size]))

                    else:
                        _ = np.matmul(self.layer_errors[t + 1][:, 1:], np.transpose(self.weights[t + 1]))
                        self.layer_errors[t] = _ * sigmoid_derivative(add_ones(self.layers[t][i:i + batch_size]))

                    # Here we compute how we should change the weights
                for t in range(len(self.delta_weights)):
                    if t == len(self.delta_weights) - 1:
                        h = np.transpose(
                            np.matmul(np.transpose(self.layer_errors[t]), self.activations[t][i:i + batch_size, :]))
                        self.delta_weights[t] += h + self.weights[t] / self.train_data.shape[0]
                    else:
                        h = np.transpose(np.matmul(np.transpose(self.layer_errors[t][:, 1:]),
                                                   self.activations[t][i:i + batch_size, :]))
                        self.delta_weights[t] += h + lmbda * self.weights[t] / self.train_data.shape[0]

                    # Here we change the weights by alpha * delta_weights
                for t in range(len(self.delta_weights)):
                    self.weights[t] -= alpha * self.delta_weights[t] / (self.train_data.shape[0] / batch_size)

            j += 1
            self.forward_propagation(self.test_data)
            sys.stdout.write(f"Epoch: {j}, error: {logistic_cost(self.activations[-1], self.test_results)}\n")
            # sys.stdout.write(self.hypothesis[0, :])
            sys.stdout.flush()

        print(mistakes(self.activations[-1], self.real_test_results))


class SVM:
    def __init__(self, train_data, test_data, train_results, test_results, weights, similarities):
        self.train_data = train_data
        self.test_data = test_data
        self.train_results = train_results
        self.test_results = test_results
        self.weights = weights
        self.similarities = similarities
        self.hypothesis = np.matmul(self.similarities, weights)

    def element_svm_positive_cost(self, k, b):
        # Create a DataFrame containing the numbers and the cost value corresponding to numbers
        hypothesis = pd.DataFrame({"Numbers": pd.Series(self.hypothesis.flatten()),
                                   "Values": pd.Series(np.zeros([self.hypothesis.shape[0]]))})

        # Here we set the values of cost for numbers in this collection
        hypothesis["Values"][hypothesis["Numbers"] >= 1] = 0
        hypothesis["Values"][hypothesis["Numbers"] < 1] = b - k * hypothesis["Numbers"]

        h = np.array(hypothesis["Values"])
        return np.reshape(h, [h.shape[0], 1])

    def element_svm_negative_cost(self, k, b):
        # Create a DataFrame containing the numbers and the cost value corresponding to numbers
        hypothesis = pd.DataFrame({"Numbers": pd.Series(self.hypothesis.flatten()),
                                   "Values": pd.Series(np.zeros([self.hypothesis.shape[0]]))})

        # Here we set the values of cost for numbers in this collection
        hypothesis["Values"][hypothesis["Numbers"] <= -1] = 0
        hypothesis["Values"][hypothesis["Numbers"] > -1] = b + k * hypothesis["Numbers"]

        h = np.array(hypothesis["Values"])
        return np.reshape(h, [h.shape[0], 1])

    def regularised_svm_cost(self, k, b):
        weights_cost = sum(self.weights) / self.weights.shape[0]

        return sum(sum(self.train_results * self.element_svm_positive_cost(k, b) + (1 - self.train_results) * self.element_svm_negative_cost(k, b))) / self.train_results.shape[0] + weights_cost

    def svm_gradient(self, k, b):
        m = pd.Series((self.train_results * self.element_svm_positive_cost(k, b) + (1 - self.train_results) * self.element_svm_negative_cost(k, b)).flatten())
        derivative = pd.Series(np.zeros([self.train_results.shape[0]]))
        h = pd.DataFrame({"Costs": m, "Results": pd.Series(self.train_results.flatten()), "Derivative": derivative})

        h["Derivative"][(h.Costs != 0.0) & (h.Results == 0.0)] = k
        h["Derivative"][(h.Costs != 0.0) & (h.Results == 1.1)] = -k

        k = np.reshape(np.array(h.Derivative), [self.train_data.shape[0], 1])
        return np.transpose(np.matmul(np.transpose(k), self.similarities))

    def svm_gradient_descent(self, k, b, threshold, alpha):
        print(self.regularised_svm_cost(k, b))
        while self.regularised_svm_cost(k, b) > threshold:
            self.weights -= self.svm_gradient(k, b) * alpha / self.train_data.shape[0]
            self.hypothesis = np.matmul(self.similarities, self.weights)
            print(self.regularised_svm_cost(k, b))

        # results = pd.DataFrame({"Hypothesis": pd.Series(self.hypothesis.flatten()), "Result": pd.Series(np.zeros([self.hypothesis.shape[0]]))})
        # results.Result[results.Hypothesis >= 1] = 1
        n = 0
        for i in range(self.hypothesis.shape[0]):

            if self.hypothesis[i][0] >= 1 and self.train_results[i][0] == 1.0:
                n += 1
            elif self.hypothesis[i][0] < 1 and self.train_results[i][0] == 0.0:
                n += 1

        print(n/self.train_results.shape[0])


