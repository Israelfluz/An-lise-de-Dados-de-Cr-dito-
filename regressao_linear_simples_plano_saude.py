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

# b1
regressor.intercept_

# b0
regressor.coef_  

# IMPORTANTODO BIBLIOTECA PARA GERAR GRAFICOS 
import matplotlib.pyplot as plt
plt.scatter(x, y)

# VISUALIZANDO E LINHA DA REGRESSÃO
plt.plot(x, regressor.predict(x), color = 'red')

# ====== CRIANDO TÍTULO PARA O GRAFICO ======
plt.title('Regressão linear simples')

# ====== CRIANDO TÍTULO PARA O EIXO X DO GRAFICO =====
plt.xlabel('Idade')

# ====== CRIANDO TÍTULO PARA O EIXO Y DO GRAFICO =====
plt.ylabel('Custo')

# ====== CALCULANDO O CUSTO DO PLANO DE SAÚDE (PREVISÃO))
previsao1 = regressor.intercept_ + regressor.coef_*[40]
previsao2 = regressor.predict([[40]])

# ==== VISUALIZANDO O VALOR DE SCORE - COMO ESTA SE COMPORTANDO O PREVISOR ====
score = regressor.score(x, y)

#==================================================================================
#== VISUALIZANDO OS VALORES RESIDUAIS QUE SÃO AS DISTÂNCIAS DOS PONTOS PARA A LINHA
#==================================================================================

# IMPORTANDO BIBLIOTECA 
from yellowbrick.regressor import ResidualsPlot
visualizador = ResidualsPlot(regressor)
visualizador.fit(x, y)
visualizador.poof()