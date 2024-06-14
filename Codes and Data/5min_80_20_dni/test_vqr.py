#implementation of 2 layer NN using qiskit

#importing libraries
import numpy as np

import pandas as pd
from qiskit.circuit.library import ZZFeatureMap, TwoLocal

import matplotlib.pyplot as plt
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B,ADAM
from qiskit_machine_learning.algorithms import QSVR, VQR
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
import pylatexenc
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import ZZFeatureMap,RealAmplitudes
from sklearn.model_selection import RepeatedStratifiedKFold
from qiskit_algorithms.utils import algorithm_globals
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

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
#loading dataset. the features are already selected
features_train =  pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Codes_RFE/5min_80_20/features_train.csv")
features_test = pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Codes_RFE/5min_80_20/features_test.csv")
target_train =  pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Codes_RFE/5min_80_20/target_train.csv")
target_test = pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Codes_RFE/5min_80_20/target_test.csv")

X_train = np.array(features_train).reshape(len(features_train), len(features_train.iloc[0]))
y_train = np.array(target_train['ghi_kt_5min'])

X_test = np.array(features_test).reshape(len(features_test),  len(features_test.iloc[0]))
y_test = np.array(target_test['ghi_kt_5min'])

#zzfm = ZZFeatureMap(nqubits, reps = 1)

feature_map = QuantumCircuit(len(features_train.iloc[0]), name="fm")

params_x = [Parameter(f"x{i+1}") for i in range(len(features_train.iloc[0]))]

#angle encoding
for i, param in enumerate(params_x):
    feature_map.ry(param, i)

#ansatz
#trying real amplitudes ansatz
# ansatz = RealAmplitudes(len(features_train.iloc[0]), reps=1)

#ansatz = QuantumCircuit(len(features_train.iloc[0]), name="vf")
#params_y = [Parameter(f"y{i+1}") for i in range(len(features_train.iloc[0]))]

nqubits = 5 #5 input parameters

twol = TwoLocal(nqubits, 'ry', 'cx', 'linear', reps = 1) #using two local as ansatz
# Change rep(etition)s above to suit your needs.

vqr = VQR(nqubits, feature_map = feature_map, ansatz = twol, loss='squared_error', 
          optimizer=L_BFGS_B(maxiter=10,ftol=10E-5,iprint=0))

vqr.fit(X_train, y_train)

vqr_hat_5min = vqr.predict(X_test)

vqr_hat_5min = vqr_hat_5min* target_test['ghi_clear_5min']

#Avaliação do desempenho: RMSE, MAE, R², MAPE
test_vqr_error = calculate_errors(vqr_hat_5min, target_test['ghi_5min'].values, 1)

print(test_vqr_error)