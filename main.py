import tensorflow
import keras
import numpy as np
import scipy as stats
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('student-mat.csv', sep=';')
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Predict G3 which is the final grade
# y = mx + b | m = derivative of slope | x = data | b = intercept of line | y = predicted value
# the column of data I would like to predict
predict = "G3"

# set up the features as an array of numbers dropping the predict column which is G3
features = np.array(data.drop([predict], 1))
# set up the label arrat as an array of data that we want to predict this is the data we will test against
labels = np.array(data[predict])

# print(labels, "\n")
# print(features, "\n")

best = 0  # variable holds the best score of the model up until now

# train the model 100 times to get the best score possible
for _ in range(10000):

    # destructurate our created model into training data and test data
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(features, labels, test_size=0.1)

    # create out linear regression model
    linear = linear_model.LinearRegression()

    # put our data into the model
    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)  # gives accuracy score / coeficient of corelation

    # compare accuaricy score to best score and if its bigger replace it
    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

# print(f'Co {linear.coef_}')  # list of m variable in multi dimensional space
# print(f'Intercept {linear.intercept_}')  # b

print(best)

pickle_in = open("studentmodel.pickle", "rb")

linear = pickle.load(pickle_in)
predict = linear.predict(x_test)

# print(data.head(0))
# print(f'Coeficient {linear.coef_}')
# print(f'Intercept {linear.intercept_}')

# for i in range(len(predict)):
#     print(predict[i], x_test[i], y_test[i])
#     # pass

p = "G1"

plt.style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()
