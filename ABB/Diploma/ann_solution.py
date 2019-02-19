# get data
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
dataset = pd.read_csv('income.csv')
# data munging
#check if there is na or null
#check_item = pd.isnull(dataset).any()
# check if there is null or na value
check = pd.isnull(dataset).any()

check_row = np.where(pd.isnull(dataset.income) ==True)
dataset = dataset.drop([17706])
check = pd.isnull(dataset).any()

# Artificial Neural Network

# Part 1 - Data Preprocessing
# Importing the dataset

X = dataset.iloc[:, 0:14].values
y = dataset.iloc[:, 14].values
 
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 2])#workclass
labelencoder_X_2 = LabelEncoder() 
X[:, 3] = labelencoder_X_2.fit_transform(X[:, 4])#education
labelencoder_X_3 = LabelEncoder() 
X[:, 5] = labelencoder_X_3.fit_transform(X[:, 6])#marital-status
labelencoder_X_4 = LabelEncoder() 
X[:, 6] = labelencoder_X_4.fit_transform(X[:, 7])#occupation
labelencoder_X_5 = LabelEncoder() 
X[:, 7] = labelencoder_X_5.fit_transform(X[:, 8])#relationship
labelencoder_X_6 = LabelEncoder() 
X[:, 8] = labelencoder_X_6.fit_transform(X[:, 9])#race
labelencoder_X_7 = LabelEncoder() 
X[:, 9] = labelencoder_X_7.fit_transform(X[:, 10])#sex
labelencoder_X_8 = LabelEncoder() 
X[:, 13] = labelencoder_X_8.fit_transform(X[:, 13])#native-country
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)#y
testLabel = X[0:10]
testAgain = pd.DataFrame(testLabel)

onehotencoder = OneHotEncoder(categorical_features = [1,3,5,6,7,8,9,13])
X = onehotencoder.fit_transform(X).toarray()
testX = X[1,:]
#onehotencoder_y = OneHotEncoder(categorical_features = [0])
#y = onehotencoder_y.fit_transform(y).toarray()
#X = X[:, 1:]
test1 = pd.DataFrame(X[1,:])
check_nan = np.isnan(X).any()



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu',input_dim = 13843))

# Adding the second hidden layer
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Predicting a single new observation
"""Predict if this person has income which is >50K:
age: 13
workclass:  State-gov
fnlwgt: 77516
education: Bachelors
education-num:13
marital-status: Never-married
occupation: Adm-clerical
relationship: Not-in-family
race: White
sex: Male
capital-gain:2174
capital-loss:0
hours-per-week:40
native-country: United-States
income:
"""
# trasnform the input

X_pre = pd.DataFrame(X[1,:])
 
labelencoder_X_1_pre = LabelEncoder()
#X_pre[:, 1] = labelencoder_X_1_pre.fit_transform(X[:, 2])#workclass
workclass= labelencoder_X_1_pre.fit([{str(val)} for val in X[:,1]]).transform([str("State-gov")])

labelencoder_X_2_pre = LabelEncoder() 
education = labelencoder_X_2_pre.fit([{str(val)} for val in X[:,3]]).transform([{str(val)} for val in X[1,3]])

labelencoder_X_3_pre = LabelEncoder() 
marital = labelencoder_X_3_pre.fit([{str(val)} for val in X[:,5]]).transform([{str(val)} for val in X[1,5]])

labelencoder_X_4_pre = LabelEncoder() 
occupation = labelencoder_X_4_pre.fit([{str(val)} for val in X[:,6]]).transform([{str(val)} for val in X[1,6]])

labelencoder_X_5_pre = LabelEncoder() 
relationship = labelencoder_X_5_pre.fit([{str(val)} for val in X[:,7]]).transform([{str(val)} for val in X[1,7]])

labelencoder_X_6_pre = LabelEncoder() 
race = labelencoder_X_6_pre.fit([{str(val)} for val in X[:,8]]).transform([{str(val)} for val in X[1,8]])

labelencoder_X_7_pre = LabelEncoder() 
sex = labelencoder_X_7_pre.fit([{str(val)} for val in X[:,9]]).transform([{str(val)} for val in X[1,9]])

labelencoder_X_8_pre = LabelEncoder() 
country = labelencoder_X_8_pre.fit([{str(val)} for val in X[:,13]]).transform([{str(val)} for val in X[1,13]])
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)#y

# combine those all
#income_input_attribute = workclass;education;marital; occupation;relationship; race; sex; country]

onehotencoder = OneHotEncoder(categorical_features = [1,3,5,6,7,8,9,13])
X = onehotencoder.fit_transform(X).toarray()

#use model to do prediction
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
#new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#Re_test = testLabel.reshape(1,-1)
onehotencoder = OneHotEncoder(categorical_features = [1,3,5,6,7,8,9,13])
testEncoder= onehotencoder.fit_transform(testLabel).toarray()
