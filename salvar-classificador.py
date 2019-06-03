import pandas as pd

# IMPORTAÇÃO DA BASE

base = pd.read_csv('credit-data.csv')

# =============================================================================
# VALORES NEGATIVOS PARA A IDADE
# =============================================================================
base.loc[base.age < 0, 'age'] = 40.92

# ===== DIVISÃO DOS ATRIBUTOS EM PREVISORES E CLASSES               
               
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# =============================================================================
# VALORES FALTANTES: Nan
# =============================================================================
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

# =============================================================================
# =============== PADRONIZAÇÃO DOS VALORES ===============
# =============================================================================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# CLASSIFICADOR COM SVM
classificadorSVM = SVC(kernel = 'rbf', C = 2.0, probability = True)

# TREINAMENTO. MÉTODO FIT, EFETIVAMENTE GERA O CLASSIFICADOR
classificadorSVM.fit(previsores, classe)

# CLASSIFICADOR COM RANDOM FOREST
classificadorRandomForest = RandomForestClassifier(n_estimators = 40, criterion = 'entropy')

# TREINAMENTO. MÉTODO FIT, EFETIVAMENTE GERA O CLASSIFICADOR
classificadorRandomForest.fit(previsores, classe)

# CLASSIFICADOR COM MLPCLSSIFIER
classificadorMLP = MLPClassifier(verbose = True, max_iter = 1000,
                                 tol = 0.000010, solver = 'adam',
                                 hidden_layer_sizes=(100), activation = 'relu',
                                 batch_size = 200, learning_rate_init = 0.001)

# TREINAMENTO. MÉTODO FIT, EFETIVAMENTE GERA O CLASSIFICADOR
classificadorMLP.fit(previsores, classe)

# IMPORTAÇÃO DA BIBLIOTECA QUE FAZ A GRAVAÇÃO DO ARQ.
import pickle

# SALVANDO O CLASSIFICADOR SVM
pickle.dump(classificadorSVM, open('svm_finalizado.sav', 'wb'))

# SALVANDO O CLASSIFICADOR RANDOM FOREST
pickle.dump(classificadorRandomForest, open('random_forest_finalizado.sav', 'wb'))

# SALVANDO O CLASSIFICADOR MLP
pickle.dump(classificadorMLP, open('mlp_finalizado.sav', 'wb'))
