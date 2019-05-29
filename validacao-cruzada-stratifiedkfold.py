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
# VALORES FALTANTES ou CORREÇÃO DELES: Nan  
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

# ESCOLHA E DEFINIÇÃO DO ALGORITMO PARA TESTE 
from sklearn.naive_bayes import GaussianNB

# IMPORTAÇÃO DA BIBLIOTECA
import numpy as np
a = np.zeros(5)

# DA O FORMATO DA VARIÁVEL PREVISORES
previsores.shape
previsores.shape[0]
b = np.zeros(shape=(previsores.shape[0], 1))

# =============================================================================
# REORGANIZANDO OS DADOS COM A STRATIFICAÇÃO
# =============================================================================
from sklearn.model_selection import StratifiedKFold

# =============================================================================
# RETORNANDO A PRECISÃO DO VALOR DO ALGORITMO
# =============================================================================
from sklearn.metrics import accuracy_score, confusion_matrix
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 3)
resultados = []
matrizes = []
for indice_treinamento, indice_teste in kfold.split(previsores,
                                                    np.zeros(shape=(previsores.shape[0], 1))):
    #print('Índice treinamento: ', indice_treinamento, 'Índice teste: ', indice_teste)
    
    # CLASSIFICADOR
    classificador = GaussianNB()
    
    # TREINAMENTO. MÉTODO FIT, EFETIVAMENTE GERA O CLASSIFICADOR
    classificador.fit(previsores[indice_treinamento], classe[indice_treinamento])
    
    # COMPARAR O RESULTADO PREVISORES COM O RESULTADO INDICE TESTE
    previsoes = classificador.predict(previsores[indice_teste])
    precisao = accuracy_score(classe[indice_teste], previsoes)
    matrizes.append(confusion_matrix(classe[indice_teste], previsoes))
    resultados.append(precisao)

matriz_final = np.mean(matrizes, axis = 0)

# CONVERSÃO 
resultados = np.asarray(resultados)

# O RESULTADO DA MÉDIA
resultados.mean()

# DESVIO PADRÃO EM RELAÇÃO A MÉDIA
resultados.std()