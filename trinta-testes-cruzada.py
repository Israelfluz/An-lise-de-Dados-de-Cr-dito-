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

# IMPORATAÇÃO DA BIBLIOTECA NUMPY
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# IMPORTAÇÃO E APLICAÇÃO DO ALGORITMO ÁRVORE DE DECISÃO
from sklearn.tree import DecisionTreeClassifier

# IMPORTAÇÃO E APLICAÇÃO DO ALGORITMO NAIVE BAYES
from sklearn.naive_bayes import GaussianNB

# IMPORTAÇÃO E APLICAÇÃO DO ALGORITMO DE REGRESSÃO LOGISTICA
from sklearn.linear_model import LogisticRegression

# IMPORTAÇÃO E APLICAÇÃO DO ALGORITMO SVM
from sklearn.svm import SVC

# IMPORTAÇÃO E APLICAÇÃO DO ALGORITMO KNN
from sklearn.neighbors import KNeighborsClassifier

# IMPORTAÇÃO E APLICAÇÃO DO ALGORITMO FLORESTA RANDÔMICA
from sklearn.ensemble import RandomForestClassifier

# IMPORTAÇÃO E APLICAÇÃO DE REDES NEURAIS
from sklearn.neural_network import MLPClassifier

resultados30 = []
for i in range(30):
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state = i)
    resultados1 = []
    for indice_treinamento, indice_teste in kfold.split(previsores, np.zeros(shape=(classe.shape[0], 1))):
        #classificador = GaussianNB()
        #classificador = DecisionTreeClassifier()
        #classificador = LogisticRegression()
        #classificador = SVC(kernel = 'rbf', random_state = 1, C = 2.0)
        #classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p = 2)
        #classificador = RandomForestClassifier(n_estimators=40, criterion='entropy', random_state=0)
        classificador = MLPClassifier(verbose = True, max_iter = 1000,
                              tol = 0.000010, solver='adam',
                              hidden_layer_sizes=(100), activation = 'relu',
                              batch_size=200, learning_rate_init=0.001)
        
        
        classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
        previsoes = classificador.predict(previsores[indice_teste])
        precisao = accuracy_score(classe[indice_teste], previsoes)
        resultados1.append(precisao)
    resultados1 = np.asarray(resultados1)
    media = resultados1.mean()
    resultados30.append(media)
    
resultados30 = np.asarray(resultados30)    
resultados30.mean()
for i in range(resultados30.size):
    print(str(resultados30[i]).replace('.', ','))

