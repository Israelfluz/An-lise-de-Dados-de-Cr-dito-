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

# =============================================================================
# =============== VALIDAÇÃO CRUZADA ========================
#==============================================================================
from sklearn.model_selection import cross_val_score

# ESCOLHA E DEFINIÇÃO DO ALGORITMO PARA TESTE 
from sklearn.naive_bayes import GaussianNB

# CLASSIFICADOR
classificador = GaussianNB()

# COMPARAR O RESULTADO PREVISORES COM O RESULTADO CLASSE_TESTE
resultados = cross_val_score(classificador, previsores, classe, cv = 10)

# O RESULTADO DA MÉDIA
resultados.mean()

# DESVIO PADRÃO EM RELAÇÃO A MÉDIA
resultados.std()