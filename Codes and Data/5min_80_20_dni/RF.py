# -*- coding: utf-8 -*-
"""
Implementation of quantum machine learning algorithm(s) for dataset benchmarking
Based on the data from https://doi.org/10.1063/1.5094494 - A comprehensive dataset for the accelerated development and benchmarking of solar forecasting methods
Data available at https://zenodo.org/records/2826939 
Created on Fri Feb 16 08:14:29 2024

@author: victor
"""
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split,RandomizedSearchCV
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import r2_score

#defining a function for return the error metrics
def calculate_errors(predicted, target, persist_nRMSE=1):

    RMSE = np.sqrt(((predicted - target) ** 2).mean())
    
    nRMSE = RMSE/target.mean()

    MAE = (abs(predicted - target)).mean()
    
    nMAE = MAE/target.mean()
    
    try:
        MAPE = (abs((predicted - target)/target)).mean()
    except:
        MAPE = 10e5
    
    MBE = (predicted - target).mean()
    
    Skill = 1 - nRMSE / persist_nRMSE
    
    #r2score = R2Score()
    #R2 = r2score(torch.tensor(predicted.values), torch.tensor(target.values))
    R2 = r2_score(target, predicted)
    
    # http://dx.doi.org/10.1002/2017SW001669
    MR = np.median(abs(np.log(predicted/target)))
    MSA = 100 * np.exp(MR) - 1
    
    # http://dx.doi.org/10.1002/2017SW001669
    MR = np.median(np.log(predicted/target))
    SSPB = 100 * np.sign(MR) * (np.exp(abs(MR)) - 1)
       
    return RMSE, nRMSE, MAE, nMAE, MAPE, MBE, Skill, R2

#loading dataset
#loading dataset. the features are already selected
features_train =  pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Codes_RFE/5min_80_20/features_train.csv")
features_test = pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Codes_RFE/5min_80_20/features_test.csv")
target_train =  pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Codes_RFE/5min_80_20/target_train.csv")
target_test = pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Codes_RFE/5min_80_20/target_test.csv")


#building the SVR model
TOC = time.perf_counter()#time benchmarking
rf_reg = RandomForestRegressor()

parameters_rf={"n_estimators": [100,200,300,400,500,600],
            "max_features": ['sqrt',1,'log2'],
            "max_depth": [50,60,70,80,90,100,110,120,130],
            "min_samples_split": [1,2,3,4,5,6,7,8,9],
            "min_samples_leaf": [1,2,3,4,5,6,7,8,9],
            "bootstrap": [True, False] }


# parameters_rf={"n_estimators": [250,300,350],
#             "max_features": [1.0, 'sqrt', 'log2'],
#             "max_depth": [90,100,110],
#             "min_samples_split": [ 3,4,5],
#             "min_samples_leaf": [3, 4,5],
#             "bootstrap": [True, False] }

# tunned_parameters=GridSearchCV(rf_reg,
#                           param_grid=parameters_rf,
#                           scoring='neg_mean_squared_error',
#                           #scoring='neg_mean_absolute_error',
#                           refit=True,
#                           #cv=TimeSeriesSplit(n_splits=5),
#                           cv=5, #[(slice(None), slice(None))], #10,
#                           n_jobs=-1,
#                           verbose=0)

tunned_parameters = RandomizedSearchCV(rf_reg,
                                  param_distributions=parameters_rf,
                                  scoring='neg_mean_squared_error',
                                  refit=True,
                                  cv=5,
                                  n_jobs=-1,
                                  verbose=0,
                                  n_iter=100,
                                  random_state=30) #setting the seed

#tuning hyperparameters with different data sizes
tunned_parameters.fit(features_train.iloc[:,:], target_train.iloc[:,:]['ghi_kt_5min']) #for ghi 5 min ahead
#tunned_parameters.fit(train_5m.iloc[:, :-3], train_5m.iloc[:,-3]) #for ghi 5 min ahead

print('\nRF regression model - best hyperparameters:')
print('V: ', tunned_parameters.best_params_)
print()

rf_reg = RandomForestRegressor(**tunned_parameters.best_params_)

#validating the results in the validation dataset
#rf_reg.fit(X_val_5m, y_val_5m['ghi_kt_5min'])

#prediction is done over the kt index
# y_pred_val_RF = rf_reg.predict(X_val_5m)

# #error calculated on the real irradiance value: I = kt * I_cs
# val_RF_error = calculate_errors(y_pred_val_RF * y_val_5m['ghi_clear_5min'].values, y_val_5m['ghi_5min'].values, 1)

#testing the model with the test dataset
rf_reg.fit(features_test, target_test['ghi_kt_5min'])

#prediction is done over the kt index
y_pred_test_rf = rf_reg.predict(features_test)

#error calculated on the real irradiance value: I = kt * I_cs
test_rf_error = calculate_errors(y_pred_test_rf * target_test['ghi_clear_5min'].values, target_test['ghi_5min'].values, 1)

print(test_rf_error)

TIC = time.perf_counter() # end clock
print(f"\nTrained the SVR model in {TIC - TOC:0.4f} seconds \n"); del TIC, TOC