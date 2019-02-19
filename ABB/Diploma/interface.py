# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 19:49:20 2018

@author: mingyux
"""

#%%
from tkinter import *
from tkinter import messagebox
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
dataset = pd.read_csv('income.csv')
#点击button时对应的操作  
def btnHelloClicked(dataset):

                age = int(e1.get())
                age = 11
                workclass = str(e2.get())
                workclass = " Private"
                fnlwgt = int(e3.get())
                fnlwgt = 11
                education = str(e4.get())
                education = " HS-grad"
                education_num = int(e5.get())
                education_num = 11
                marital = str(e6.get())
                marital = " Never-married"
                occupation = str(e7.get())
                occupation = " Sales"
                relationship = str(e8.get())
                relationship = " Husband"
                race = str(e9.get())
                race = " White"
                sex = str(e10.get())
                sex = " Female"
                gain = float(e11.get())
                gain = 0
                loss = float(e12.get())
                loss = 0
                hours = int(e13.get())
                hours = 40
                native = str(e14.get())
                native = " United-States"
            # master.messagebox.showinfo(cd)
                da=np.array([age,workclass,fnlwgt,education,education_num,marital,occupation,relationship, race,sex,gain,loss,hours,native])
            #process the input 
                input_new  = pd.DataFrame(da.reshape(1,14),columns=['age', 'workclass', 'fnlwgt', 'education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country'])  
                from transform import trans
                result_row = trans(input_new,dataset)
                
            
                messagebox.showinfo(title='prediction', message=workclass+" "+str(age))
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

button1 = Button(master, text='submit',command = btnHelloClicked(dataset))
button1.grid(row=14, column=1)

#button2 = Button(master, text='Zoom out')
#button2.grid(row=2, column=3)

mainloop()