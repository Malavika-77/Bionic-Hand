import numpy as np       #This line imports the NumPy library and assigns it the alias np. NumPy is a popular library in Python for working with arrays and matrices.
import pandas as pd      #This line imports the Pandas library and assigns it the alias pd. Pandas is a popular library in Python for working with data frames and data manipulation.
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler    # This line imports the MinMaxScaler class from the preprocessing module of the scikit-learn library. MinMaxScaler is a type of data normalization technique used to scale features to a given range, typically between 0 and 1.

sc = MinMaxScaler(feature_range=(0, 1)) #This line creates a MinMaxScaler object named sc with a feature range of 0 to 1.

#print(tf.VERSION)
#print(tf.keras.__version__)

#reads in four CSV files and assigns them to four separate Pandas data frames named
# rock_dataset, scissors_dataset, paper_dataset, and ok_dataset.
#Each file is assumed to have no header row and contains a matrix of numerical values representing a particular hand gesture,

#The header=None argument specifies that the data file does not have a header row

rock_dataset = pd.read_csv("0.csv", header=None)  # class = 0
scissors_dataset = pd.read_csv("1.csv", header=None)  # class = 1
paper_dataset = pd.read_csv("2.csv", header=None)  # class = 2
ok_dataset = pd.read_csv("3.csv", header=None)  # class = 3

#concatenates the four data frames (rock_dataset, scissors_dataset, paper_dataset, and ok_dataset)
# vertically using pd.concat() method to create a single data frame named dataset

frames = [rock_dataset, scissors_dataset, paper_dataset, ok_dataset]
dataset = pd.concat(frames)

#iloc indexing method to select the rows in dataset_train using the shuffled indices from the previous step.
# This effectively randomizes the order of the rows in dataset_train.
#it calls the reset_index() method with drop=True argument to reset the index of dataset_train without adding a new column,
# so that the index is a continuous sequence of integers starting from 0.

dataset_train = dataset.iloc[np.random.permutation(len(dataset))]
dataset_train.reset_index(drop=True)

# two lists X_train and y_train, which will be used to store the features and labels of the training set, respectively.
X_train = []
y_train = []

#It then loops through each row of the shuffled dataset_train data frame using the range() function
# and the shape[0] attribute to determine the number of rows in the data frame.

#For each row, it selects a slice of the data frame using the iloc method with i:1 + i to select the ith row and the 0:64 to select the first 64 columns of the row.
# This creates a 2D NumPy array named row with dimensions (1, 64)

#t then uses the np.reshape() function to convert the 2D array row into a 1D array with dimensions (64, 1).
# This reshapes the input features from a 1x64 matrix into a 64x1 column vector.

#t selects the last column of the ith row of the dataset_train data frame using iloc with -1: to select the last column.
# This represents the label of the current example.

#Finally, it appends the reshaped input features row and the corresponding label to the X_train and y_train lists, respectively.

for i in range(0, dataset_train.shape[0]):
    row = np.array(dataset_train.iloc[i:1 + i, 0:64].values)
    X_train.append(np.reshape(row, (64, 1)))
    y_train.append(np.array(dataset_train.iloc[i:1 + i, -1:])[0][0])

#Converts the X_train and y_train lists into NumPy arrays using np.array().

X_train = np.array(X_train)
y_train = np.array(y_train)

# Reshape to one flatten vector
#Reshapes the X_train array into a 1D array using reshape(), so that each feature vector is a row in the array.
#Applies feature scaling to the X_train array using the MinMaxScaler from scikit-learn, which scales each feature to a value between 0 and 1

X_train = X_train.reshape(X_train.shape[0] * X_train.shape[1], 1)
X_train = sc.fit_transform(X_train)

#Reshapes the X_train array back into a 3D array with dimensions (-1, 8, 8), where -1 indicates that the size of the first dimension is inferred from the other dimensions.
# This reshaping converts each flattened feature vector into an 8x8 matrix.
# Reshape again after normalization to (-1, 8, 8)
X_train = X_train.reshape((-1, 8, 8))

#Converts the y_train array into a one-hot encoded array using np.eye(), where each row represents a label and each column represents a class.
# The element in each row and column is 1 if the example belongs to that class, and 0 otherwise.
# Convert to one hot

y_train = np.eye(np.max(y_train) + 1)[y_train]

#Prints the shape of the training data (X_train and y_train), the test data (X_test and y_test), and the full data (X_train and y_train combined).

print("All Data size X and y")
print(X_train.shape)
print(y_train.shape)


#Splits the data into training and testing sets using indexing.
# The first 7700 examples are used for training, and the remaining examples are used for testing.
# Splitting Train/Test


X_test = X_train[7700:]
y_test = y_train[7700:]
print("Test Data size X and y")
print(X_test.shape)
print(y_test.shape)

X_train = X_train[0:7700]
y_train = y_train[0:7700]
print("Train Data size X and y")
print(X_train.shape)
print(y_train.shape)

# Creating the model
#This code builds and trains a LSTM neural network classifier using the Keras library

#Import the necessary Keras layers and the Sequential model from Keras.

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Create a new instance of the Sequential model, which will be used to build the LSTM network.

classifier = Sequential()

#Add four LSTM layers to the network using add(), each with 50 units and a dropout rate of 0.2 to prevent overfitting.

classifier.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 8)))
classifier.add(Dropout(0.2))

classifier.add(LSTM(units=50, return_sequences=True))
classifier.add(Dropout(0.2))

classifier.add(LSTM(units=50, return_sequences=True))
classifier.add(Dropout(0.2))

classifier.add(LSTM(units=50))
classifier.add(Dropout(0.2))

#Add three dense layers to the network using add(), with 64, 128, and 4 units, respectively.
# The last layer uses a softmax activation function to output probabilities for each of the four classes.

classifier.add(Dense(units=64))
classifier.add(Dense(units=128))

classifier.add(Dense(units=4, activation="softmax"))

#Compile the model using the Adam optimizer and binary cross-entropy loss function.

classifier.compile(optimizer='adam', loss='binary_crossentropy')

#Train the model using the fit() function, with the training data (X_train and y_train) and a batch size of 32.
# The model is trained for 250 epochs, and the training progress is displayed with a verbose level of 2.

classifier.fit(X_train, y_train, epochs=250, batch_size=32, verbose=2)

# Save
classifier.save("model_cross_splited_data.h5")
print("Saved model to disk")

###############################################

from tensorflow import keras
# # Load Model
# model = keras.models.load_model('model_cross_splited_data.h5')
# model.summary()
###################################################
#defines a function called evaluateModel that takes two arguments prediction and y,
# which are the predicted labels and the true labels, respectively.

#function evaluates the classification accuracy of the model and returns the percentage of correct classifications.
#he for loop iterates over each label in y, compares the predicted label with the true label using np.array_equal, and counts the number of correct classifications.
# The final accuracy is calculated by dividing the number of correct classifications by the total number of labels and multiplying the result by 100.

def evaluateModel(prediction, y):
    good = 0
    for i in range(len(y)):
       # if (prediction[i] == np.argmax(y[i])):\
       if np.array_equal(prediction[i], np.argmax(y[i])):
           # do something

           good = good + 1
    return (good / len(y)) * 100.0

#he result_test and result_train variables store the predicted labels for the test and train data, respectively,
# using the predict method of the trained classifier mode

#The evaluateModel function is then called with result_test and y_test as arguments
# to evaluate the classification accuracy on the test data, and with result_train and y_train as arguments
# to evaluate the classification accuracy on the train data.
# The classification accuracy results and the predicted labels are printed to the console.

result_test = classifier.predict(X_test)
print("Correct classification rate on test data")
print(evaluateModel(result_test, y_test))
print('result=',result_test)

#max_indices = np.argmax(result_test, axis=1)
#print(max_indices)


result_train = classifier.predict(X_train)
print("Correct classification rate on train data")
print(evaluateModel(result_train, y_train))
print('result=',result_train)