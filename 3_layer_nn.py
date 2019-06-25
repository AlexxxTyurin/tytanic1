import numpy as np
import pandas as pd
from tools import *
import sys


class NN:
    def __init__(self, input_data, actual_results):
        self.input_data = input_data
        self.actual_results = actual_results

        self.weights_1 = np.array(pd.read_csv("/Users/alextyurin/Desktop/pycharm_projects/tytanic/weights1.csv"))
        self.weights_2 = np.array(pd.read_csv("/Users/alextyurin/Desktop/pycharm_projects/tytanic/weights2.csv"))
        self.weights_3 = np.array(pd.read_csv("/Users/alextyurin/Desktop/pycharm_projects/tytanic/weights3.csv"))

        self.hidden_layer1 = np.zeros(np.matmul(self.input_data, self.weights_1).shape)
        self.hidden_layer1_activation = np.zeros(add_ones(self.hidden_layer1).shape)

        self.hidden_layer2 = np.zeros(np.matmul(self.hidden_layer1_activation, self.weights_2).shape)
        self.hidden_layer2_activation = np.zeros(add_ones(self.hidden_layer2).shape)

        self.output_layer = np.zeros(np.matmul(self.hidden_layer2_activation, self.weights_3).shape)
        self.hypothesis = np.zeros(self.output_layer.shape)

        self.delta_weights_1 = np.zeros(np.transpose(self.weights_1).shape)
        self.delta_weights_2 = np.zeros(np.transpose(self.weights_2).shape)
        self.delta_weights_3 = np.zeros(np.transpose(self.weights_3).shape)

    def error(self):
        return sum(sum((self.hypothesis - self.actual_results) ** 2)) / (2 * self.hypothesis.shape[0])

    def forward_propagation(self):
        self.hidden_layer1 = np.matmul(self.input_data, self.weights_1)
        self.hidden_layer1_activation = add_ones(sigmoid(self.hidden_layer1))

        self.hidden_layer2 = np.matmul(self.hidden_layer1_activation, self.weights_2)
        self.hidden_layer2_activation = add_ones(sigmoid(self.hidden_layer2))

        self.output_layer = np.matmul(self.hidden_layer2_activation, self.weights_3)
        self.hypothesis = sigmoid(self.output_layer)

    def bp_gradient_descent(self, epochs, alpha, lmbda):
        for i in range(epochs):
            self.forward_propagation()
            sys.stdout.write(f"Epoch: {i}, error: {logistic_cost(self.hypothesis, self.actual_results)}\n")
            sys.stdout.flush()
            for j in range(0, self.hypothesis.shape[0], 33):
                self.delta_weights_1 = np.zeros(np.transpose(self.weights_1).shape)
                self.delta_weights_2 = np.zeros(np.transpose(self.weights_2).shape)
                self.delta_weights_3 = np.zeros(np.transpose(self.weights_3).shape)

                # Find the error on unsigmoid data
                output_error = self.hypothesis[j:j+33] - self.actual_results[j:j+33]

                # Find the error in the second hidden layer
                _ = np.matmul(output_error, np.transpose(self.weights_3))
                hidden_layer2_error = _ * sigmoid_derivative(add_ones(self.hidden_layer2[j:j+33]))

                # Find the error in the first hidden layer
                _ = np.matmul(hidden_layer2_error[:, 1:], np.transpose(self.weights_2))
                hidden_layer1_error = _ * sigmoid_derivative(add_ones(self.hidden_layer1[j:j + 33]))

                # print(np.transpose(output_error).shape, self.hidden_layer_activation[j:j+81].shape)
                # print(np.transpose(hidden_layer_error[:, 1:]).shape, self.input_data[j:j+81].shape)
                self.delta_weights_3 = self.delta_weights_3 + np.matmul(np.transpose(output_error), self.hidden_layer2_activation[j:j + 33]) + lmbda * np.transpose(self.weights_3) / self.hypothesis.shape[0]
                self.delta_weights_2 = self.delta_weights_2 + np.matmul(np.transpose(hidden_layer2_error[:, 1:]), self.hidden_layer1_activation[j:j+33]) + lmbda * np.transpose(self.weights_2) / self.hypothesis.shape[0]
                self.delta_weights_1 = self.delta_weights_1 + np.matmul(np.transpose(hidden_layer1_error[:, 1:]), self.input_data[j:j + 33]) + lmbda * np.transpose(self.weights_1) / self.hypothesis.shape[0]

            self.weights_3 -= np.transpose(self.delta_weights_3) * alpha / 27
            self.weights_2 -= np.transpose(self.delta_weights_2) * alpha / 27
            self.weights_1 -= np.transpose(self.delta_weights_1) * alpha / 27


if __name__ == "__main__":
    b = pd.read_csv("/Users/alextyurin/Desktop/pycharm_projects/tytanic/data.csv")
    actual_results = np.array(b["Survived"], dtype="int").reshape((891, 1))

    input_data = np.array(b)
    input_data[:, 0] = 1
    input_data[:, 3] = (input_data[:, 3] - mean(input_data[:, 3])) / (variance(input_data[:, 3]) ** 0.5)
    input_data[:, 6] = (input_data[:, 6] - mean(input_data[:, 6])) / (variance(input_data[:, 6]) ** 0.5)
    input_data[:, 7] = (input_data[:, 7] - mean(input_data[:, 7])) / (variance(input_data[:, 7]) ** 0.5)

    a = NN(input_data, actual_results)
    # a.forward_propagation()

    a.bp_gradient_descent(4000, 0.1, 0.1)

    for i in range(20):
        print(f"Real value: {actual_results[i]}, prediction: {a.hypothesis[i]}")

    print(mistakes(a.hypothesis, actual_results))
    print(a.hypothesis.shape[0])

