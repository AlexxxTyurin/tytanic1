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

        self.hidden_layer = np.zeros(np.matmul(self.input_data, self.weights_1).shape)
        self.hidden_layer_activation = np.zeros(add_ones(np.matmul(self.input_data, self.weights_1)).shape)

        self.output_layer = np.zeros(np.matmul(self.hidden_layer_activation, self.weights_2).shape)
        self.hypothesis = np.zeros(self.output_layer.shape)

        self.delta_weights_1 = np.zeros(np.transpose(self.weights_1).shape)
        self.delta_weights_2 = np.zeros(np.transpose(self.weights_2).shape)

    def error(self):
        return sum(sum((self.hypothesis - self.actual_results) ** 2)) / (2 * self.hypothesis.shape[0])

    def forward_propagation(self):
        self.hidden_layer = np.matmul(self.input_data, self.weights_1)
        self.hidden_layer_activation = add_ones(sigmoid(self.hidden_layer))

        self.output_layer = np.matmul(self.hidden_layer_activation, self.weights_2)
        self.hypothesis = sigmoid(self.output_layer)

    def bp_gradient_descent(self, epochs, alpha, lmbda):
        for i in range(epochs):
            self.forward_propagation()
            sys.stdout.write(f"Epoch: {i}, error: {logistic_cost(self.hypothesis, self.actual_results)}\n")
            sys.stdout.flush()

            self.delta_weights_1 = np.zeros(np.transpose(self.weights_1).shape)
            self.delta_weights_2 = np.zeros(np.transpose(self.weights_2).shape)

            # Find the error on unsigmoid data
            output_error = self.hypothesis - self.actual_results

            # Find the error in the hidden layer
            _ = np.matmul(output_error, np.transpose(self.weights_2))
            hidden_layer_error = _ * sigmoid_derivative(add_ones(self.hidden_layer))

            # print(np.transpose(output_error).shape, self.hidden_layer_activation[j:j+81].shape)
            # print(np.transpose(hidden_layer_error[:, 1:]).shape, self.input_data[j:j+81].shape)
            self.delta_weights_2 = self.delta_weights_2 + np.matmul(np.transpose(output_error), self.hidden_layer_activation) + lmbda * np.transpose(self.weights_2) / self.hypothesis.shape[0]
            self.delta_weights_1 = self.delta_weights_1 + np.matmul(np.transpose(hidden_layer_error[:, 1:]), self.input_data) + lmbda * np.transpose(self.weights_1) / self.hypothesis.shape[0]

            self.weights_2 -= np.transpose(self.delta_weights_2) * alpha
            self.weights_1 -= np.transpose(self.delta_weights_1) * alpha


if __name__ == "__main__":
    b = pd.read_csv("/Users/alextyurin/Desktop/pycharm_projects/tytanic/data.csv")
    actual_results = np.array(b["Survived"], dtype="int").reshape((714, 1))

    input_data = np.array(b)
    input_data[:, 0] = 1
    input_data[:, 3] = (input_data[:, 3] - mean(input_data[:, 3])) / (variance(input_data[:, 3]) ** 0.5)
    input_data[:, 6] = (input_data[:, 6] - mean(input_data[:, 6])) / (variance(input_data[:, 6]) ** 0.5)
    input_data[:, 7] = (input_data[:, 7] - mean(input_data[:, 7])) / (variance(input_data[:, 7]) ** 0.5)

    a = NN(input_data, actual_results)
    # a.forward_propagation()

    a.bp_gradient_descent(40000, 0.001, 0.4)

    for i in range(20):
        print(f"Real value: {actual_results[i]}, prediction: {a.hypothesis[i]}")

    print(mistakes(a.hypothesis, actual_results))

    # This part is for implementing our updated weights for testing set
    right_weights1 = a.weights_1
    right_weights2 = a.weights_2
    #
    # # In this part we import the testing set and add bias unit
    # test = pd.read_csv("/Users/alextyurin/Desktop/pycharm_projects/tytanic/test.csv")
    # test = pd.DataFrame({"Pclass": test["Pclass"],
    #                      "Sex": test["Sex"],
    #                      "Age": test["Age"],
    #                      "SibSp": test["SibSp"],
    #                      "Parch": test["Parch"],
    #                      "Ticket": test["Ticket"],
    #                      "Fare": test["Fare"],
    #                      "Embarked": test["Embarked"]})
    #
    # test["Sex"][test["Sex"] == "male"] = 1
    # test["Sex"][test["Sex"] == "female"] = 0
    # test["Age"][test["Age"].isnull()] = np.average(test["Age"][test["Age"].notnull()])
    # for i in range(test.shape[0]):
    #     test["Ticket"][i] = str(test["Ticket"][i]).split(" ")[-1]
    # test["Embarked"][test["Embarked"] == "C"] = 0
    # test["Embarked"][test["Embarked"] == "Q"] = 1
    # test["Embarked"][test["Embarked"] == "S"] = 2
    #
    # test = np.array(test)
    # test = add_ones(test)
    # print(test.shape)
    # #
    # # In this part we just forward propagate. Actually, the same things are written in forward propagation method for
    # # NN class
    # test_hidden_layer = np.matmul(test, right_weights1)
    # test_hidden_layer_activation = add_ones(sigmoid(test_hidden_layer))
    # #
    # test_output_layer = np.matmul(test_hidden_layer_activation, right_weights2)
    # test_hypothesis = sigmoid(test_output_layer)
    # #
    # # In order to get the accuracy of prediction for the testing set, I have to send the submission file to Kaggle
    # submission = pd.Series([result(el) for el in test_hypothesis])
    # submission = pd.DataFrame({"PassengerId": np.arange(892, 1310), "Survived": submission})
    # submission.to_csv("/Users/alextyurin/Desktop/pycharm_projects/tytanic/submission.csv", index=False)