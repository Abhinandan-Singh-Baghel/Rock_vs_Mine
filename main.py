import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# loading the dataset to a pandas dataframe

sonar_data = pd.read_csv('sonar_data.csv', header=None)

# print(sonar_data.head())
# print(sonar_data.shape)

# print(sonar_data.describe())

# print(sonar_data[60].value_counts())

# csv is stored in array according to the column


X = sonar_data.drop(columns = 60, axis=1)
Y = sonar_data[60]

# print(X)
# print(Y)


#Training and Test Data 

X_train, X_test, Y_train, Y_test = train_test_split(X , Y, test_size = 0.1, stratify= Y, random_state=1)

print(X.shape, X_train.shape, X_test.shape)

# print(X_train)
# print(Y_train)

model = LogisticRegression()

#training the Logistic Regression model with the training data

model.fit(X_train, Y_train)

#accuracy on training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on training data : ', training_data_accuracy)

#accuracy on test data

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on test data : ', test_data_accuracy)


