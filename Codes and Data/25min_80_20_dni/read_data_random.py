# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 18:14:06 2024

@author: volive04
"""

#Importing libraries
import pandas as pd
import gc
from sklearn.model_selection import train_test_split

#reading the datasets
#datasets have values for B, V and L for 5-,10-,15-,20-, and 30-minutes spanning from 2014 to 2016
irradiance = pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Data/Irradiance_features_intra-hour.csv")
sky_image = pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Data/Sky_image_features_intra-hour.csv")
target = pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Data/Target_intra-hour.csv")

#creating dataset for forecasting irradiance 5-,10-,15-,20-, and 30-minutes ahead

irradiance_5m = irradiance[['timestamp','B(ghi_kt|5min)', 'V(ghi_kt|5min)', 'L(ghi_kt|5min)']]
sky_image_5m = sky_image.drop(columns=['AVG(NRB)', 'STD(NRB)', 'ENT(NRB)', 'timestamp'])
target_5m = target[['ghi_5min','dni_5min','ghi_clear_5min', 'dni_clear_5min','ghi_kt_5min','dni_kt_5min']]
df_5m = pd.concat([irradiance_5m, sky_image_5m,target_5m], axis = 1).dropna() #merging the features and the target df for 5 min and removing NAs

irradiance_10m = irradiance[['timestamp','B(ghi_kt|10min)', 'V(ghi_kt|10min)', 'L(ghi_kt|10min)']]
# = sky_image.drop(columns=['AVG(NRB)', 'STD(NRB)', 'ENT(NRB)', 'timestamp'])
target_10m = target[['ghi_10min','dni_10min','ghi_clear_10min', 'dni_clear_10min','ghi_kt_10min','dni_kt_10min']]
df_10m = pd.concat([irradiance_10m, sky_image_5m,target_10m], axis = 1).dropna() #merging the features and the target df for 10 min and removing NAs

irradiance_15m = irradiance[['timestamp','B(ghi_kt|15min)', 'V(ghi_kt|15min)', 'L(ghi_kt|15min)']]
# = sky_image.drop(columns=['AVG(NRB)', 'STD(NRB)', 'ENT(NRB)', 'timestamp'])
target_15m = target[['ghi_15min','dni_15min','ghi_clear_15min', 'dni_clear_15min','ghi_kt_15min','dni_kt_15min']]
df_15m = pd.concat([irradiance_15m, sky_image_5m,target_15m], axis = 1).dropna() #merging the features and the target df for 15 min and removing NAs

irradiance_20m = irradiance[['timestamp','B(ghi_kt|20min)', 'V(ghi_kt|20min)', 'L(ghi_kt|20min)']]
# = sky_image.drop(columns=['AVG(NRB)', 'STD(NRB)', 'ENT(NRB)', 'timestamp'])
target_20m = target[['ghi_20min','dni_20min','ghi_clear_20min', 'dni_clear_20min','ghi_kt_20min','dni_kt_20min']]
df_20m = pd.concat([irradiance_20m, sky_image_5m,target_20m], axis = 1).dropna() #merging the features and the target df for 20 min and removing NAs

irradiance_25m = irradiance[['timestamp','B(ghi_kt|25min)', 'V(ghi_kt|25min)', 'L(ghi_kt|25min)']]
# = sky_image.drop(columns=['AVG(NRB)', 'STD(NRB)', 'ENT(NRB)', 'timestamp'])
target_25m = target[['ghi_25min','dni_25min','ghi_clear_25min', 'dni_clear_25min','ghi_kt_25min','dni_kt_25min']]
df_25m = pd.concat([irradiance_25m, sky_image_5m,target_25m], axis = 1).dropna() #merging the features and the target df for 25 min and removing NAs

irradiance_30m = irradiance[['timestamp','B(ghi_kt|30min)', 'V(ghi_kt|30min)', 'L(ghi_kt|30min)']]
# = sky_image.drop(columns=['AVG(NRB)', 'STD(NRB)', 'ENT(NRB)', 'timestamp'])
target_30m = target[['ghi_30min','dni_30min','ghi_clear_30min', 'dni_clear_30min','ghi_kt_30min','dni_kt_30min']]
df_30m = pd.concat([irradiance_30m, sky_image_5m,target_30m], axis = 1).dropna() #merging the features and the target df for 30 min and removing NAs

del irradiance, sky_image,target, irradiance_5m,irradiance_10m,irradiance_15m,irradiance_20m,irradiance_25m,irradiance_30m
del sky_image_5m,target_5m,target_10m,target_15m,target_20m,target_25m,target_30m

#splitting training and validation datasets randomly: kt_ghi
X_train_5m, X_temp, y_train_5m, y_temp = train_test_split(df_5m.iloc[:, 1:-6], df_5m[['ghi_kt_5min','ghi_clear_5min','ghi_5min']], test_size=0.2, random_state=3003)
X_val_5m, X_test_5m, y_val_5m, y_test_5m = train_test_split(X_temp, y_temp, test_size=0.5, random_state=3003)

X_train_10m, X_temp, y_train_10m, y_temp = train_test_split(df_10m.iloc[:, 1:-6], df_10m[['ghi_kt_10min','ghi_clear_10min','ghi_10min']], test_size=0.2, random_state=3003)
X_val_10m, X_test_10m, y_val_10m, y_test_10m = train_test_split(X_temp, y_temp, test_size=0.5, random_state=3003)

X_train_15m, X_temp, y_train_15m, y_temp = train_test_split(df_15m.iloc[:, 1:-6], df_15m[['ghi_kt_15min','ghi_clear_15min','ghi_15min']], test_size=0.2, random_state=3003)
X_val_15m, X_test_15m, y_val_15m, y_test_15m = train_test_split(X_temp, y_temp, test_size=0.5, random_state=3003)

X_train_20m, X_temp, y_train_20m, y_temp = train_test_split(df_20m.iloc[:, 1:-6], df_20m[['ghi_kt_20min','ghi_clear_20min', 'ghi_20min']], test_size=0.2, random_state=3003)
X_val_20m, X_test_20m, y_val_20m, y_test_20m = train_test_split(X_temp, y_temp, test_size=0.5, random_state=3003)

X_train_25m, X_temp, y_train_25m, y_temp = train_test_split(df_25m.iloc[:, 1:-6], df_25m[['ghi_kt_25min','ghi_clear_25min','ghi_25min']], test_size=0.2, random_state=3003)
X_val_25m, X_test_25m, y_val_25m, y_test_25m = train_test_split(X_temp, y_temp, test_size=0.5, random_state=3003)

X_train_30m, X_temp, y_train_30m, y_temp = train_test_split(df_30m.iloc[:, 1:-6], df_30m[['ghi_kt_30min','ghi_clear_30min','ghi_30min']], test_size=0.2, random_state=3003)
X_val_30m, X_test_30m, y_val_30m, y_test_30m = train_test_split(X_temp, y_temp, test_size=0.5, random_state=3003)


del X_temp,y_temp

#cleaning memory
gc.collect()






