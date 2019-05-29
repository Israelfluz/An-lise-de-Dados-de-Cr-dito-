import pandas as pd
import numpy as np

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

# =============================================================================
# ============= CRIAÇÃO DA BASE DE TREINAMENTO E BASE DE TESTE ================
# =============================================================================
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

# IMPORTAÇÃO DA BIBLIOTECA
from tensorflow import keras
#from keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# CLASSIFICADOR
classificador = keras.models.Sequential()

# PRIMEIRA CAMADA OCULTA
classificador.add(Dense(units = 2, activation = 'relu', input_dim = 3))

# OUTRA CAMADA OCULTA:
classificador.add(Dense(units = 2, activation = 'relu'))

# CAMADA DE SAÍDA:
classificador.add(Dense(units = 1, activation = 'sigmoid'))

# COMPILAR A REDE
classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# TREINAMENTO. MÉTODO FIT, EFETIVAMENTE GERA O CLASSIFICADOR
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 10, nb_epoch = 100)

# COMPARAR O RESULTADO PREVISORES COM O RESULTADO CLASSE_TESTE
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)
