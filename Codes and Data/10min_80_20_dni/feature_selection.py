# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 18:14:06 2024

@author: volive04
"""

#loading libraries
import numpy as np

from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
#from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV,RFE
import pandas as pd

#reading the datasets
#datasets have values for B, V and L for 5-,10-,15-,20-, and 30-minutes spanning from 2014 to 2016
irradiance = pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Data/Irradiance_features_intra-hour.csv")
sky_image = pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Data/Sky_image_features_intra-hour.csv")
target = pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Data/Target_intra-hour.csv")


irradiance_10m = irradiance[['timestamp','B(dni_kt|10min)', 'V(dni_kt|10min)', 'L(dni_kt|10min)']]
sky_image_10m = sky_image.drop(columns=['AVG(NRB)', 'STD(NRB)', 'ENT(NRB)', 'timestamp'])
target_10m = target[['dni_10min','dni_10min','dni_clear_10min', 'dni_clear_10min','dni_kt_10min','dni_kt_10min']]
df_10m = pd.concat([irradiance_10m, sky_image_10m,target_10m], axis = 1).dropna() #merging the features and the target df for 5 min and removing NAs
df_10m = df_10m.drop(columns = ['timestamp'])
#splitting training and validation datasets randomly: kt_dni
features_train, features_test, target_train, target_test = train_test_split(df_10m.iloc[:, 0:-6], df_10m[['dni_kt_10min','dni_clear_10min','dni_10min']], test_size=0.2, shuffle=False)


#------------------------------------------------------------------------------
#Feature selection implementation using RFE
#RFE - source:https://machinelearningmastery.com/rfe-feature-selection-in-python/ 
#feature selection is important for reduce computational processing time during the quantum algorithms implementations
#the selected features will be used for both classical and quantum models
#------------------------------------------------------------------------------
rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=5)
model = DecisionTreeRegressor()
pipeline = Pipeline(steps=[('s',rfe),('m',model)])

#model evaluation
cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
n_scores = cross_val_score(pipeline, features_train, target_train['dni_kt_10min'],
                           scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')

#reporting performance
#print('MAE: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

#fitting the model
rfe.fit(features_train, target_train['dni_kt_10min'])

#selecting the feature
for i in range(features_train.shape[1]):
 print('Column: %d, Selected %s, Rank: %.3f' % (i, rfe.support_[i], rfe.ranking_[i]))
 
list(features_train.columns) #the selected features were B, V, L, ENT(G) and ENT(RB)

#selecting the features from the dataset
indices = np.where(rfe.ranking_[:] == 1)[0]
features_train_2 = features_train.iloc[:, indices]
features_test_2 = features_test.iloc[:, indices]

#exporting the dataframes
features_train_2.to_csv('features_train.csv', index=False)
features_test_2.to_csv('features_test.csv', index=False)
target_train.to_csv('target_train.csv', index=False)
target_test.to_csv('target_test.csv', index=False)


