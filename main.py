import numpy as np
import pandas as pd
from tools import *
import sys
import math
import gc


def similarity(input_data, sigma):
    input_data = np.array([input_data[i, :] for _ in range(input_data.shape[0]) for i in range(input_data.shape[0])])
    landmarks = np.array([input_data[i, :] for i in range(input_data.shape[0]) for _ in range(input_data.shape[0])])
    print(input_data)
    print(landmarks)
    # res = np.exp(((input_data-landmarks) ** 2) / (-2 * sigma))
    # print(res)

    # np.array([input_data[i, :] for _ in range(input_data.shape[0]) for i in range(input_data.shape[0])])
    # return np.reshape(res, [input_data.shape[0], input_data.shape[0]])


if __name__ == "__main__":
    b = pd.read_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/train.csv")
    actual_results = np.array(b["Survived"], dtype="float")
    actual_results = np.reshape(actual_results, [len(actual_results), 1])

    # Set new features:sex_male and sex_female. If person is a female, then sex_male = 0 and sex_female = 1
    sex_male = pd.Series(b["Sex"] == "male", dtype="float")
    sex_female = pd.Series(b["Sex"] == "female", dtype="float")

    # Ages of some passengers are missed, that is why we fill these missed cells with an average value
    b["Age"][b["Age"].isnull()] = np.average(np.array(b["Age"][b["Age"].notnull()]))

    # Here we set 0 for the rows with missed Cabin and 1 with an existing Cabin
    b["Cabin"][b["Cabin"].notnull()] = 1
    b["Cabin"][b["Cabin"].isnull()] = 0

    # Here we set new features: cherbourg, queenstown and southampton. People from southampton have southampton feature
    #  equal to 1 while other features are equal to 0.
    cherbourg = pd.Series(b["Embarked"] == "C", dtype="float")
    queenstown = pd.Series(b["Embarked"] == "Q", dtype="float")
    southampton = pd.Series(b["Embarked"] == "S", dtype="float")

    # Here we create a new DataFrame with new features
    b = pd.DataFrame({"Bias_unit": b["Survived"],
                      "Pclass": b["Pclass"],
                      "Sex_male": sex_male,
                      "Sex_female": sex_female,
                      "Age": b["Age"],
                      "SibSp": b["SibSp"],
                      "Parch": b["Parch"],
                      "Fare": b["Fare"],
                      "Cabin": b["Cabin"],
                      "Cherbourg": cherbourg,
                      "Queenstown": queenstown,
                      "Southampton": southampton})

    # Create the numpy array from the DataFrame object
    input_data = np.array(b, dtype="float")

    # Here we set the bias unit to be equal to 1
    input_data[:, 0] = 1.0

    # Here we standardize the Age and Fare
    input_data[:, 4] = (input_data[:, 4] - mean(input_data[:, 4])) / (variance(input_data[:, 4]) ** 0.5)
    input_data[:, 7] = (input_data[:, 7] - mean(input_data[:, 7])) / (variance(input_data[:, 7]) ** 0.5)

    # Here we divide our data into train_data and test_data
    train_data = input_data[0:int(0.8 * input_data.shape[0]), :]
    test_data = input_data[int(0.8 * input_data.shape[0]):, :]

    train_results = actual_results[0:int(0.8 * actual_results.shape[0]), :]
    test_results = actual_results[int(0.8 * actual_results.shape[0]):, :]

    # Here we generate the list of weights
    weights = np.random.sample([train_data.shape[0], 1])

    # Here we calculate the similarities via a procedure. Firstly we create 2 vectors: data and landmark
    data = np.array([train_data[i, :] for _ in range(train_data.shape[0]) for i in range(train_data.shape[0])])
    landmark = np.array([train_data[i, :] for i in range(train_data.shape[0]) for _ in range(train_data.shape[0])])

    # Secondly, we calculate the similarities vector and then we reshape it
    similarities = np.array([sum(np.exp(((data[i, :] - landmark[i, :]) ** 2) / (-2 * 4))) for i in range(data.shape[0])])
    similarities = np.reshape(similarities, [train_data.shape[0], train_data.shape[0]])

    a = SVM(train_data, test_data, train_results, test_results, weights, similarities)
    # print(a.hypothesis)
    print(a.regularised_svm_cost(3, 3))








