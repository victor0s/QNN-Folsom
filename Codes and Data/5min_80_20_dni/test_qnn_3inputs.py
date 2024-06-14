# -*- coding: utf-8 -*-
"""
Trabalhando com o conjuntode dados de Folsom para aplicação dos modelos QCNN, NBEATS e GMDH
Autor:Felipe Pinto Marinho
Data:08/01/2024
"""

#Carregando alguns pacotes relevantes
#Para o QCNN
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit_machine_learning.algorithms import QSVR
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor
import pylatexenc
import pennylane as qml
from sklearn.svm import SVR
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_algorithms.utils import algorithm_globals

#Para ML
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
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

#Para o GMDH
import pandas as pd
import seaborn as sns
from gmdh import Combi, split_data

#LightGBM
import lightgbm as lgb
import optuna as opt


#Carregando o dataset para análise
irradiance = pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Data/Irradiance_features_intra-hour.csv")
sky_image = pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Data/Sky_image_features_intra-hour.csv")
target = pd.read_csv("C:/Users/volive04/Desktop/UoG/24WINTER/Quantum Computing/Project/Data/Target_intra-hour.csv")

irradiance_15m = irradiance[['timestamp','B(ghi_kt|15min)', 'V(ghi_kt|15min)', 'L(ghi_kt|15min)']]
sky_image_15m = sky_image.drop(columns=['AVG(NRB)', 'STD(NRB)', 'ENT(NRB)', 'timestamp'])
target_15m = target[['ghi_15min','dni_15min','ghi_clear_15min', 'dni_clear_15min','ghi_kt_15min','dni_kt_15min']]
df_15m = pd.concat([irradiance_15m, sky_image_15m,target_15m], axis = 1).dropna() #merging the features and the target df for 5 min and removing NAs
df_15m = df_15m.drop(columns = ['timestamp'])
#splitting training and validation datasets randomly: kt_ghi
features_train, features_test, target_train, target_test = train_test_split(df_15m.iloc[:, 0:-6], df_15m[['ghi_kt_15min','ghi_clear_15min','ghi_15min']], test_size=0.2, shuffle=False)


#Construindo um fature map simples
X_train = np.array(features_train[['B(ghi_kt|15min)', 'V(ghi_kt|15min)', 'L(ghi_kt|15min)']]).reshape(len(features_train), 1)
y_train = np.array(target_train['ghi_kt_15min'])

X_test = np.array(features_test[['B(ghi_kt|15min)', 'V(ghi_kt|15min)', 'L(ghi_kt|15min)']]).reshape(len(features_test), 1)
y_test = np.array(target_test['ghi_kt_15min'])


feature_map = QuantumCircuit(len(features_train.iloc[0]), name="fm")

params_x = [Parameter(f"x{i+1}") for i in range(len(features_train.iloc[0]))]

#angle encoding
for i, param in enumerate(params_x):
    feature_map.ry(param, i)

#ansatz
#trying real amplitudes ansatz
# ansatz = RealAmplitudes(len(features_train.iloc[0]), reps=1)

ansatz = QuantumCircuit(len(features_train.iloc[0]), name="vf")
params_y = [Parameter(f"y{i+1}") for i in range(len(features_train.iloc[0]))]

for i, param in enumerate(params_y):
    ansatz.ry(param, i)

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

VQR_hat_15min = VQR_hat_15min * target_test['ghi_clear_15min']


VQR_hat_15min = VQR_hat_15min * target_test['ghi_clear_15min']
print("O valor do RMSE é:", np.sqrt(mean_squared_error(target_test['ghi_15min'], VQR_hat_15min)))
print("O valor do MAE é:", mean_absolute_error(target_test['ghi_15min'], VQR_hat_15min))
print("O valor do R² é:", r2_score(target_test['ghi_15min'], VQR_hat_15min))
print("O valor do MAPE é:", mean_absolute_percentage_error(target_test['ghi_15min'], VQR_hat_15min))
    