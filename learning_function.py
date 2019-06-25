import pandas as pd
import numpy as np
import operator

train = pd.read_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/train.csv")
test = pd.read_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/test.csv")
data = pd.read_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/data.csv")

# data["Embarked"][train["Embarked"] == "C"] = 0
# print(train[train["Age"].isnull()].shape)
data = data[train["Age"].notnull()]
data.to_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/data.csv", index=False)

train = pd.DataFrame({"Pclass": train["Pclass"], "Name": train["Name"], "Sex": train["Sex"], "Age": train["Age"],
                      "SibSp": train["SibSp"], "Parch": train["Parch"],"Ticket": train["Ticket"], "Fare": train["Fare"],
                      "Embarked": train["Embarked"]})

test = pd.DataFrame({"Pclass": test["Pclass"], "Name": test["Name"], "Sex": test["Sex"], "Age": test["Age"],
                      "SibSp": test["SibSp"], "Parch": test["Parch"],"Ticket": test["Ticket"], "Fare": test["Fare"],
                      "Embarked": test["Embarked"]})
all_passengers = pd.read_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/all_passengers.csv")

southampton = all_passengers[all_passengers["Embarked"] == "S"]
cherbourg = all_passengers[all_passengers["Embarked"] == "C"]
queenstown = all_passengers[all_passengers["Embarked"] == "Q"]

rich = all_passengers[all_passengers["Fare"] > 500]
average = all_passengers[operator.and_(all_passengers["Fare"] <= 500, all_passengers["Fare"] > 50)]
poor = all_passengers[50 <= all_passengers["Fare"]]

first_class = all_passengers[all_passengers["Pclass"] == 1]
second_class = all_passengers[all_passengers["Pclass"] == 2]
third_class = all_passengers[all_passengers["Pclass"] == 3]

young = all_passengers[all_passengers["Age"] < 16]
moderate = all_passengers[operator.and_(all_passengers["Age"] >= 16, all_passengers["Age"] < 30)]
middle = all_passengers[operator.and_(all_passengers["Age"] >= 30, all_passengers["Age"] < 50)]
old = all_passengers[all_passengers["Age"] > 50]

southampton.to_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/southampton.csv")
cherbourg.to_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/cherbourg.csv")
queenstown.to_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/queenstown.csv")

rich.to_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/rich.csv")
average.to_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/average.csv")
poor.to_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/poor.csv")

first_class.to_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/first_class.csv")
second_class.to_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/second_class.csv")
third_class.to_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/third_class.csv")

young.to_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/young.csv")
moderate.to_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/moderate.csv")
middle.to_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/middle.csv")
old.to_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/old.csv")






# all_passengers.to_csv("/Users/vladikturin/Desktop/pycharm_projects/tytanic/all_passengers.csv", index=False)


# print(test)



