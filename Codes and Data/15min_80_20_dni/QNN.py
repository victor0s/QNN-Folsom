# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:44:01 2024

@author: volive04
"""

#VQE multiple predictors


#Carregando alguns pacotes relevantes
#Para o QCNN
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap,TwoLocal
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B,ADAM
from qiskit_machine_learning.algorithms import QSVR
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
import pylatexenc
#import pennylane as qml
from sklearn.svm import SVR
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_algorithms.utils import algorithm_globals

#Para ML
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from qiskit.circuit.library import ZZFeatureMap,RealAmplitudes
from sklearn.preprocessing import MaxAbsScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
import sklearn
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from qiskit_algorithms.utils import algorithm_globals

import time

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
features_train =  pd.read_csv("C:/Users/victo/Desktop/UoG/24Winter/Quantum Computing/Project/Codes_RFE/15min_80_20_dni/features_train.csv")
features_test = pd.read_csv("C:/Users/victo/Desktop/UoG/24Winter/Quantum Computing/Project/Codes_RFE/15min_80_20_dni/features_test.csv")
target_train =  pd.read_csv("C:/Users/victo/Desktop/UoG/24Winter/Quantum Computing/Project/Codes_RFE/15min_80_20_dni/target_train.csv")
target_test = pd.read_csv("C:/Users/victo/Desktop/UoG/24Winter/Quantum Computing/Project/Codes_RFE/15min_80_20_dni/target_test.csv")


#------------------------------------------------------------------------------
#Implementação do Variational Quantum Regressor (VQR)
#VQR - source:https://qiskit-community.github.io/qiskit-machine-learning/tutorials/02_neural_network_classifier_and_regressor.html 
#------------------------------------------------------------------------------
#Construindo um fature map simples
#X_train = np.array(features_train[['B(dni_kt|15min)', 'V(dni_kt|15min)', 'L(dni_kt|15min)']]).reshape(len(features_train), 3)
TOC = time.perf_counter()#time benchmarking

#qiskit seed; for reproducibility
algorithm_globals.random_seed = 30

X_train = np.array(features_train).reshape(len(features_train), len(features_train.iloc[0]))
y_train = np.array(target_train['dni_kt_15min'])

X_test = np.array(features_test).reshape(len(features_test),  len(features_test.iloc[0]))
y_test = np.array(target_test['dni_kt_15min'])

feature_map = QuantumCircuit(len(features_train.iloc[0]), name="fm")

params_x = [Parameter(f"x{i+1}") for i in range(len(features_train.iloc[0]))]

#angle encoding
for i, param in enumerate(params_x):
    feature_map.ry(param, i)

#ansatz
#trying real amplitudes ansatz
# ansatz = RealAmplitudes(len(features_train.iloc[0]), reps=1)

ansatz = TwoLocal(5, 'ry', 'cx', 'linear', reps = 1)

# ansatz = QuantumCircuit(len(features_train.iloc[0]), name="vf")
# params_y = [Parameter(f"y{i+1}") for i in range(len(features_train.iloc[0]))]

# for i, param in enumerate(params_y):
#     ansatz.ry(param, i)

#Construindo um circuito
qc = QNNCircuit(feature_map=feature_map, ansatz=ansatz)
qc.draw()
#Construindo a QNN
regression_estimator_qnn = EstimatorQNN(circuit=qc)

objective_func_vals = [] #to store the values of the objective function

def callback_graph(weights, obj_func_eval):
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    #plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()


regressor = NeuralNetworkRegressor(
    neural_network=regression_estimator_qnn,
    loss="squared_error",
    #optmizer = COBYLA(maxiter=10, tol=10E-5)
    optimizer= L_BFGS_B(maxiter=10,ftol=10E-5,iprint=0), #plot indicates that the model coverges after 4 iterations
    callback=callback_graph,
)

plt.rcParams["figure.figsize"] = (12, 6)

X_train.shape
np.array(y_train)

regressor.fit(X_train, y_train)
VQR_hat_15min = regressor.predict(X_test)
VQR_hat_15min = np.squeeze(VQR_hat_15min)

VQR_hat_15min = VQR_hat_15min * target_test['dni_clear_15min']

test_VQR_error = calculate_errors(VQR_hat_15min.values, target_test['dni_15min'].values, 1)

print(test_VQR_error)

TIC = time.perf_counter() # end clock
print(f"\nTrained the XGB model in {TIC - TOC:0.4f} seconds \n"); del TIC, TOC