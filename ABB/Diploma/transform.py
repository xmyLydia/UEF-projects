# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 23:02:07 2018

@author: mingyux
"""
def trans(input_new,dataset):
    input_new[['age']] = input_new[['age']].astype(int)
    input_new[['fnlwgt']] = input_new[['fnlwgt']].astype(int)
    input_new[['education-num']] = input_new[['education-num']].astype(int)
    input_new[['capital-gain']] = input_new[['capital-gain']].astype(int)
    input_new[['capital-loss']] = input_new[['capital-loss']].astype(int)
    input_new[['hours-per-week']] = input_new[['hours-per-week']].astype(int)
    for var in ['workclass']: #,'Name','Ticket']:
                    new = list(set(input_new[var]) - set(dataset[var]))
                    input_new.ix[input_new[var].isin(new), var] = -999
    import numpy as np
    input_new['sex'] = dataset['sex'].map({' Male':0,' Female':1})
    input_new['race'] =dataset['race'].map({' White':0,' Black':1,' Asian-Pac-Islander':2,' Amer-Indian-Eskimo':3,' Other':4})
    input_new['relationship'] =dataset['relationship'].map({' Not-in-family':0,' Husband':1,' Wife':2,' Own-child':3,' Unmarried':4,' Other-relative':5})
    
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    SubWorkclassTrans = mlb.fit([{str(val)} for val in dataset['workclass'].values]).transform([{str(val)} for val in input_new['workclass'].values])
    SubEducationTrans = mlb.fit([{str(val)} for val in dataset['education'].values]).transform([{str(val)} for val in input_new['education'].values])
    SubMaritalTrans = mlb.fit([{str(val)} for val in dataset['marital-status'].values]).transform([{str(val)} for val in input_new['marital-status'].values])
    SubOccupationTrans = mlb.fit([{str(val)} for val in dataset['occupation'].values]).transform([{str(val)} for val in input_new['occupation'].values])
    SubNativeTrans = mlb.fit([{str(val)} for val in dataset['native-country'].values]).transform([{str(val)} for val in input_new['native-country'].values])
    input_new =input_new.drop(['workclass','education','marital-status','occupation','native-country'], axis=1)
    input_sub_new = np.hstack((input_new.values,SubWorkclassTrans,SubEducationTrans,SubMaritalTrans,SubOccupationTrans,SubNativeTrans))
    return input_sub_new