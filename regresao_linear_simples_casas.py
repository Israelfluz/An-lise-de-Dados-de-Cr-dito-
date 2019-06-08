# ====== IMPORTAÇÃO DA BIBLIOTECA =======
import pandas as pd

# ====== IMPORTAÇÃO DA BASE DE DADOS =======
base = pd.read_csv('house-prices.csv')

# VARIÁVEL X QUE SÃO OS ATRIBUTOS PREVISORES (tamanho dos terrenos) 
x = base.iloc[:, 5:6].values

# VARIÁVEL Y QUE A BASE DE RESPOSTA (O valor das casas)
y = base.iloc[:, 2].values
# =============================================================================
# ============= CRIAÇÃO DA BASE DE TREINAMENTO(70%) E BASE DE TESTE(30%) ================
# =============================================================================
from sklearn.model_selection import train_test_split
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)

# ====== MODELO DE REGRESSÃO LINEAR ======
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# ====== GERANDO O TREINAMENTO ========
regressor.fit(X_treinamento, y_treinamento)

# ==== VISUALIZANDO O VALOR DE SCORE - COMO ESTA SE COMPORTANDO O PREVISOR (SCORE BAIXO NESSE) ====
score = regressor.score(X_treinamento, y_treinamento)

# ===== IMPORTANTODO BIBLIOTECA PARA GERAR GRAFICOS E VISUALIZANDO O GRAFICO COM A BASE DE DADOS TREINAMENTO ====== 
## ==== PELO GRÁFICO SE PODE VER QUE O ALGORITMO NÃO SE ADAPTOU ====
import matplotlib.pyplot as plt
plt.scatter(X_treinamento, y_treinamento)
plt.plot(X_treinamento, regressor.predict(X_treinamento), color = 'red')

# ======== RELIZANDO ALGUMAS PREVISÕES =========
previsoes = regressor.predict(X_teste)

# ======= VISUALIZANDO A DIFERENÇA DE VALORES (PREÇO DAS CASAS) =========
resultado = abs(y_teste - previsoes)

# ======= VISUALIZADNO A MÉDIA =========
resultado.mean()

# ======= OUTRA MANEIRA DE SE VISUALIZAR A DIFERENÇA ========
from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_teste, previsoes)
mse = mean_squared_error(y_teste, previsoes)

# ===== VISUALIZANDO O GRAFICO COM A BASE DE DADOS TESTE ====== 
plt.scatter(X_teste, y_teste)
plt.plot(X_teste, regressor.predict(X_teste), color = 'red')

# ====== VISUALIZANDO O SCORE DA BASE DE DADOS TESTE ===========
    regressor.score(X_teste, y_teste)