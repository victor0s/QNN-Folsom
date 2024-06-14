# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 13:54:50 2024

@author: volive04
"""
#importing libraries
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import r2_score
import xgboost

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

#loading dataset. the features are already selected
features_train =  pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Codes_RFE/15min_80_20/features_train.csv")
features_test = pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Codes_RFE/15min_80_20/features_test.csv")
target_train =  pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Codes_RFE/15min_80_20/target_train.csv")
target_test = pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Codes_RFE/15min_80_20/target_test.csv")

#------------------------------------------------------------------------------
#implementing the Light Gradient Boosting Method (lightGBM)
#Documentation: https://lightgbm.readthedocs.io/en/latest/Python-Intro.html
#------------------------------------------------------------------------------
#building the SVR model
TOC = time.perf_counter()#time benchmarking
rf_reg = RandomForestRegressor()

xgboost_reg = xgboost.XGBRegressor()

parameters={"learning_rate": [0.0001,0.0003,0.0005, 0.001,0.003,0.005,0.01,0.03,0.05],
            "n_estimators": [100,200,300,400,500,600,700],
            "max_depth": [1,2,3,4,5,6,7,8,9],
            "min_child_weight": [10E-5, 5*10E-5,10E-4,5*10E-4,10E-3],
            "gamma":[10E-4,5*10E-4,10E-3, 5*10E-3],
            "colsample_bytree":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
            "subsample":[0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75]}


# tuning_model=GridSearchCV(xgboost_reg,
#                           param_grid=parameters,
#                           scoring='neg_mean_squared_error',
#                           #scoring='neg_mean_absolute_error',
#                           refit=True,
#                           #cv=TimeSeriesSplit(n_splits=5),
#                           cv=5, #[(slice(None), slice(None))], #10,
#                           n_jobs=-1,
#                           verbose=0)

tuning_model = RandomizedSearchCV(xgboost_reg,
                                  param_distributions=parameters,
                                  scoring='neg_mean_squared_error',
                                  refit=True,
                                  cv=5,
                                  n_jobs=-1,
                                  verbose=0,
                                  n_iter=100,
                                  random_state=30) #setting the seed

tuning_model.fit(features_train.iloc[:,:], target_train.iloc[:,:]['ghi_kt_15min'])

print('\nXGBoost regression model - best hyperparameters:')
print('V: ', tuning_model.best_params_)
print()

xgboost_reg = xgboost.XGBRegressor(**tuning_model.best_params_)

# xgboost_reg.fit(X_val_15m, y_val_15m['ghi_kt_15min'])

# y_pred_val_XGB = xgboost_reg.predict(X_val_15m)

# val_XGB_error = calculate_errors(y_pred_val_XGB * y_val_15m['ghi_clear_15min'].values, y_val_15m['ghi_15min'].values, 1)

xgboost_reg.fit(features_test, target_test['ghi_kt_15min'])

y_pred_test_XGB = xgboost_reg.predict(features_test)

test_XGB_error = calculate_errors(y_pred_test_XGB * target_test['ghi_clear_15min'].values, target_test['ghi_15min'].values, 1)

print(test_XGB_error)

TIC = time.perf_counter() # end clock
print(f"\nTrained the XGB model in {TIC - TOC:0.4f} seconds \n"); del TIC, TOC