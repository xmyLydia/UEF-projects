# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 18:19:29 2018

@author: mingyux
"""
#%%
# import data and library
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
dataset = pd.read_csv('income.csv')
#just use as test

 
input = dataset[0:1]
input_new_ori = input.drop(['income'], axis=1)
#above
# check if there is null value exists 
check = pd.isnull(dataset).any()

check_row = np.where(pd.isnull(dataset.income) ==True)
dataset = dataset.drop([17706])
check_then = pd.isnull(dataset).any()

# check types
types = dataset.dtypes
# check for categorical variables, how many unique values they have
for cat in [ 'workclass', 'education', 'marital-status', 'occupation','relationship','race','sex','native-country']:
    print("Number of levels in category '{0}': \b {1:2.2f} ".format(cat, dataset[cat].unique().size))
#We then code these 3 variables who have few levels manually into numerical values.  
dataset['sex'] = dataset['sex'].map({' Male':0,' Female':1})
dataset['race'] =dataset['race'].map({' White':0,' Black':1,' Asian-Pac-Islander':2,' Amer-Indian-Eskimo':3,' Other':4})
dataset['relationship'] =dataset['relationship'].map({' Not-in-family':0,' Husband':1,' Wife':2,' Own-child':3,' Unmarried':4,' Other-relative':5})
dataset['income'] = dataset['income'].map({' <=50K':0,' >50K':1})
# encode other categorical variables as digit using Scikit-learn's MultiLabelBinarizer and treat them as new features.
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
workclass_trans = mlb.fit_transform([{str(val)} for val in dataset['workclass'].values])
education_trans = mlb.fit_transform([{str(val)} for val in dataset['education'].values])
marital_trans = mlb.fit_transform([{str(val)} for val in dataset['marital-status'].values])
occupation_trans = mlb.fit_transform([{str(val)} for val in dataset['occupation'].values])
native_trans = mlb.fit_transform([{str(val)} for val in dataset['native-country'].values])
 
# class variables
y = dataset['income'].values
# drop unused features
income_new = dataset.drop(['workclass','education','marital-status','occupation','native-country','income'], axis=1)
# add new features
income_new = np.hstack((income_new.values ,workclass_trans, education_trans,marital_trans, occupation_trans, native_trans))

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(income_new, y, test_size = 0.2, random_state = 0)

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
classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu',input_dim = 97))

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
# calculate the accuracy of predicting the test result
#%%
sum_acc = 0

for i in range(0,3541):
    if(y_pred[i]==True):
        test_y = 1
    else:
        test_y = 0
    if(y_test[i]==test_y):
        sum_acc = sum_acc+1
accuracy = sum_acc/3542
#%%
# make prediction on given data
#The most important step here is to check for new levels in the categorical variables of the submission dataset that are absent in the training set.
#We identify them and set them to our placeholder value of '-999',
seeTypes = input_new_ori.dtypes

age = int(11)
workclass = " Private"
fnlwgt = int(11)
education = " HS-grad"
education_num =int( 11)
marital = " Never-married"
occupation = " Sales"
relationship = " Husband"
race = " White"
sex = " Female"
gain =int( 0)
loss = int(0)
hours = int(40)
native = " United-States"
da=np.array([age,workclass,fnlwgt,education,education_num,marital,occupation,relationship, race,sex,gain,loss,hours,native])
input_new  = pd.DataFrame(da.reshape(1,14),columns=['age', 'workclass', 'fnlwgt', 'education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country'])  
input_new[['age']] = input_new[['age']].astype(int)
input_new[['fnlwgt']] = input_new[['fnlwgt']].astype(int)
input_new[['education-num']] = input_new[['education-num']].astype(int)
input_new[['capital-gain']] = input_new[['capital-gain']].astype(int)
input_new[['capital-loss']] = input_new[['capital-loss']].astype(int)
input_new[['hours-per-week']] = input_new[['hours-per-week']].astype(int)
seeTypes2 = input_new.dtypes
for var in ['workclass']: #,'Name','Ticket']:
    new = list(set(input_new[var]) - set(dataset[var]))
    input_new.ix[input_new[var].isin(new), var] = -999

input_new['sex'] = input_new['sex'].map({' Male':0,' Female':1})
input_new['race'] = input_new['race'].map({' White':0,' Black':1,' Asian-Pac-Islander':2,' Amer-Indian-Eskimo':3,' Other':4})
input_new['relationship'] = input_new['relationship'].map({' Not-in-family':0,' Husband':1,' Wife':2,' Own-child':3,' Unmarried':4,' Other-relative':5})

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
SubWorkclassTrans = mlb.fit([{str(val)} for val in dataset['workclass'].values]).transform([{str(val)} for val in input_new['workclass'].values])
SubEducationTrans = mlb.fit([{str(val)} for val in dataset['education'].values]).transform([{str(val)} for val in input_new['education'].values])
SubMaritalTrans = mlb.fit([{str(val)} for val in dataset['marital-status'].values]).transform([{str(val)} for val in input_new['marital-status'].values])
SubOccupationTrans = mlb.fit([{str(val)} for val in dataset['occupation'].values]).transform([{str(val)} for val in input_new['occupation'].values])
SubNativeTrans = mlb.fit([{str(val)} for val in dataset['native-country'].values]).transform([{str(val)} for val in input_new['native-country'].values])
input_new =input_new.drop(['workclass','education','marital-status','occupation','native-country'], axis=1)

# Form the new submission data set
input_sub_new = np.hstack((input_new.values,SubWorkclassTrans,SubEducationTrans,SubMaritalTrans,SubOccupationTrans,SubNativeTrans))
#np.any(np.isnan(input_sub_new))
# predict
submission = classifier.predict(input_sub_new)
submission = (submission > 0.5)
#da=np.array([1,"gov",3,4])
#df_new  = pd.DataFrame(da.reshape(1,4),columns=['age', 'workclass', 'fnlwgt', 'education'])
#df  = pd.DataFrame(da.reshape(1,14),columns=['age', 'workclass', 'fnlwgt', 'education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country'])  
#df = pd.DataFrame(data,columns=['age','class','b','d'])
#%%
# Interface
from tkinter import *
from tkinter import messagebox
#点击button时对应的操作  
def btnHelloClicked():

                age = int(e1.get())
               # age = 11
                workclass = " "+str(e2.get()).strip()
               # workclass = " Private"
                fnlwgt = int(e3.get())
              #  fnlwgt = 11
                education = " "+str(e4.get()).strip()
              #  education = " HS-grad"
                education_num = int(e5.get())
              #  education_num = 11
                marital = " "+str(e6.get()).strip()
              #  marital = " Never-married"
                occupation = " "+str(e7.get()).strip()
            #    occupation = " Sales"
                relationship = " "+str(e8.get()).strip()
              #  relationship = " Husband"
                race = " "+str(e9.get()).strip()
             #   race = " White"
                sex = " "+str(e10.get()).strip()
             #   sex = " Female"
                gain = int(e11.get())
               # gain = 0
                loss = int(e12.get())
              #  loss = 0
                hours = int(e13.get())
              #  hours = 40
                native = " "+str(e14.get()).strip()
              #  native = " United-States"
            # master.messagebox.showinfo(cd)
                da=np.array([age,workclass,fnlwgt,education,education_num,marital,occupation,relationship, race,sex,gain,loss,hours,native])
            #process the input 
                input_new  = pd.DataFrame(da.reshape(1,14),columns=['age', 'workclass', 'fnlwgt', 'education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country'])  
                from transform import trans
                input_sub_new = trans(input_new,dataset)
 # predict
                submission = classifier.predict(input_sub_new)
                submission = (submission > 0.5)
                result = submission
                if(submission==False):
                    result = "<=50K"
                else:
                    if(submission == True):
                     result = ">50K"
                messagebox.showinfo(title='prediction', message=str(result))
               # labelHello.config(text = "%.2f°C = %.2f°F" %(cd, cd*1.8+32))    
master = Tk()
var = IntVar()

Label(master, text="age").grid(sticky=E)
Label(master, text="workclass").grid(sticky=E)
Label(master, text="fnlwgt").grid(sticky=E)
Label(master, text="education").grid(sticky=E)
Label(master, text="education-num").grid(sticky=E)
Label(master, text="maritual-status").grid(sticky=E)
Label(master, text="occupation").grid(sticky=E)
Label(master, text="relationship").grid(sticky=E)
Label(master, text="race").grid(sticky=E)
Label(master, text="sex").grid(sticky=E)
Label(master, text="capital-gain").grid(sticky=E)
Label(master, text="capital-loss").grid(sticky=E)
Label(master, text="hours-per-week").grid(sticky=E)
Label(master, text="native-country").grid(sticky=E)
#labelHello = Label(master, text="Result").grid(sticky=E)
e1 = Entry(master)# age
e2 = Entry(master)#workclass
e3 = Entry(master)# age
e4 = Entry(master)#workclass
e5 = Entry(master)# age
e6 = Entry(master)#workclass
e7 = Entry(master)# age
e8 = Entry(master)#workclass
e9 = Entry(master)# age
e10 = Entry(master)#workclass
e11 = Entry(master)# age
e12 = Entry(master)#workclass
e13 = Entry(master)# age
e14 = Entry(master)#workclass
#e3 = Entry(master)
e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=3, column=1)
e5.grid(row=4, column=1)
e6.grid(row=5, column=1)
e7.grid(row=6, column=1)
e8.grid(row=7, column=1)
e9.grid(row=8, column=1)
e10.grid(row=9, column=1)
e11.grid(row=10, column=1)
e12.grid(row=11, column=1)
e13.grid(row=12, column=1)
e14.grid(row=13, column=1)
#e3.grid(row = 2, column = 1)
#checkbutton = Checkbutton(master, text='Preserve aspect', variable=var)
#checkbutton.grid(columnspan=2, sticky=W)

photo = PhotoImage(file='giphy.gif')
label = Label(image=photo)
label.image = photo
label.grid(row=0, column=2, columnspan=2, rowspan=14, sticky=W+E+N+S, padx=5, pady=5)

button1 = Button(master, text='submit',command = btnHelloClicked)
button1.grid(row=14, column=1)

#button2 = Button(master, text='Zoom out')
#button2.grid(row=2, column=3)

mainloop()

#from interface import interface_gui
#res = interface_gui(dataset)
#res 