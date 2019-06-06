# ====== IMPORTAÇÃO DA BIBLIOTECA =======
import pandas as pd

# ====== IMPORTAÇÃO DA BASE DE DADOS =======
base = pd.read_csv('plano-saude.csv')

#================================================================
# ======= SEPARANDO AS VARIÁVEIS X E Y ======
#================================================================

# VARIÁVEL X QUE SÃO OS ATRIBUTOS PREVISORES OU INDEPENDENTES (idades) 
x = base.iloc[:, 0].values

# VARIÁVEL Y QUE A BASE DE RESPOSTA (preços)
y = base.iloc[:, 1].values 

#======================================================================
#== REALIZAR O TESTE DE CORRELAÇÃO PARA VERIFICAR SE EXISTE PROXIMIDADE
#======================================================================

# ========== IMPORTAÇÃO DA BIBLIOTÉCA ==========
import numpy as np
correlação = np.corrcoef(x, y)

# ==== MUDANDO FORMATO DE UMA VARIÁVEL QUE ESTA COMO VETOR  PARA MATRIZ =====
# ==== ALGORITMOS NO SCIKIT-LEARN TEM QUE ESTAR NO FORMATO MATRIZ ====
x = x.reshape(-1,1)
  
# ====== MODELO DE REGRESSÃO LINEAR ======
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x, y)

# b0
regressor.intercept_

# b1
regressor.coef_  