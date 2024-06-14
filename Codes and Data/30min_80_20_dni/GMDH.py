# -*- coding: utf-8 -*-
"""
Trabalhando com o conjuntode dados de Folsom para aplicação dos modelos QCNN, e GMDH
Autor:Felipe Pinto Marinho
Data:08/01/2024
"""

#loading libraries
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import pandas as pd
from gmdh import Combi

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
features_train =  pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Codes_RFE/30min_80_20/features_train.csv")
features_test = pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Codes_RFE/30min_80_20/features_test.csv")
target_train =  pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Codes_RFE/30min_80_20/target_train.csv")
target_test = pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Codes_RFE/30min_80_20/target_test.csv")

#------------------------------------------------------------------------------
#Implementing GMDH model
#Documentation: https://gmdh.net/index.html
#------------------------------------------------------------------------------
model_GMDH = Combi()
model_GMDH.fit(np.array(features_train), np.array(target_train['ghi_kt_30min']))
GMDH_hat_30min = model_GMDH.predict(features_test)

#Conversão em inrradiância utilizando a irradiância de céu claro
GMDH_hat_30min = GMDH_hat_30min * target_test['ghi_clear_30min']

#Avaliação do desempenho: RMSE, MAE, R², MAPE
test_GMDH_error = calculate_errors(GMDH_hat_30min, target_test['ghi_30min'].values, 1)

print(test_GMDH_error)











