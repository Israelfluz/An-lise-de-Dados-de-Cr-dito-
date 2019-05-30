# IMPORTAÇÃO DAS BIBLIOTECAS
import pandas as pd
import pickle
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
import numpy as np

# IMPORTAÇÃO DA BASE DE DADOS
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
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

# =============================================================================
# =============== PADRONIZAÇÃO DOS VALORES ===============
# =============================================================================
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# CRIANDO VARIÁVEIS DOS CLASSIFICADORES SALVOS E ABRINDO COM BIBLIOTECA QUE FAZ A GRAVAÇÃO DO ARQ.
svm = pickle.load(open('svm_finalizado.sav', 'rb'))
random_forest = pickle.load(open('random_forest_finalizado.sav', 'rb'))
mlp = pickle.load(open('mlp_finalizado.sav', 'rb'))

# TESTE DOS CLASSIFICADORES NA MESMA BASE DE DADOS. (NÃO É AVALIAÇÃO)
resultado_svm = svm.score(previsores, classe)
resultado_random_forest = random_forest.score(previsores, classe)
resultado_mlp = mlp.score(previsores, classe)


novo_registro = [[50000, 40, 5000]]
novo_registro = np.asarray(novo_registro)
novo_registro = novo_registro.reshape(-1, 1)
novo_registro = scaler.fit_transform(novo_registro)
novo_registro = novo_registro.reshape(-1, 3)

# REALIZANDO UMA PREVISÃO DOS CLASSIFICADORES
resposta_svm = svm.predict(novo_registro)
resposta_random_forest = random_forest.predict(novo_registro)
resposta_mlp = mlp.predict(novo_registro)
