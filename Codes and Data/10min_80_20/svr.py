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
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import r2_score

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
features_train =  pd.read_csv("C:/Users/victo/Desktop/UoG/24Winter/Quantum Computing/Project/Codes_RFE/10min_80_20/features_train.csv")
features_test = pd.read_csv("C:/Users/victo/Desktop/UoG/24Winter/Quantum Computing/Project/Codes_RFE/10min_80_20/features_test.csv")
target_train =  pd.read_csv("C:/Users/victo/Desktop/UoG/24Winter/Quantum Computing/Project/Codes_RFE/10min_80_20/target_train.csv")
target_test = pd.read_csv("C:/Users/victo/Desktop/UoG/24Winter/Quantum Computing/Project/Codes_RFE/10min_80_20/target_test.csv")

#building the SVR model
TOC = time.perf_counter()#time benchmarking
svr_model = SVR()

# parameters_svr={'C': [0.1, 1, 10,100,1000],
#             'gamma': ['auto', 'scale', 0.0001, 0.001, 0.01, 0.1, 1],
#             'kernel': ['linear', 'poly','rbf', 'sigmoid']}#, 'linear']}#, 'poly', 'precomputed'


parameters_svr={'C': [0.1,1,10,100],
            'gamma': ['auto', 'scale', 0.0001, 0.001, 0.01, 0.1, 1],
            'kernel': ['linear', 'poly','rbf', 'sigmoid']}#, 'linear']}#, 'poly', 'precomputed'


# tuning_model=GridSearchCV(svr_model,
#                           param_grid=parameters_svr,
#                           scoring='neg_mean_squared_error', #parameter selection based on MSE
#                           #scoring='neg_mean_absolute_error',
#                           refit=True,
#                           cv=5, #cross-validation
#                           n_jobs=-1, #use all processors for calculations
#                           verbose=0)

tuning_model = RandomizedSearchCV(svr_model,
                                  param_distributions=parameters_svr,
                                  scoring='neg_mean_squared_error',
                                  refit=True,
                                  cv=5,
                                  n_jobs=-1,
                                  verbose=0,
                                  n_iter=5,
                                  random_state=30) #setting the seed

tuning_model.fit(features_train.iloc[:,:], target_train.iloc[:,:]['ghi_kt_10min'])

print('\nSVR regression model - best hyperparameters:')
print('V: ', tuning_model.best_params_)
print()

svr_model = SVR(**tuning_model.best_params_)

# xgboost_reg.fit(X_val_10m, y_val_10m['ghi_kt_10min'])

# y_pred_val_XGB = xgboost_reg.predict(X_val_10m)

# val_XGB_error = calculate_errors(y_pred_val_XGB * y_val_10m['ghi_clear_10min'].values, y_val_10m['ghi_10min'].values, 1)

svr_model.fit(features_test, target_test['ghi_kt_10min'])

y_pred_test_XGB = svr_model.predict(features_test)

test_XGB_error = calculate_errors(y_pred_test_XGB * target_test['ghi_clear_10min'].values, target_test['ghi_10min'].values, 1)

print(test_XGB_error)

TIC = time.perf_counter() # end clock
print(f"\nTrained the XGB model in {TIC - TOC:0.4f} seconds \n"); del TIC, TOC