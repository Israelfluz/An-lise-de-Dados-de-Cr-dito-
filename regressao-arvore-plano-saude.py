# ====== importação da biblioteca =======
import pandas as pd

# ===== carregando a base de dados a ser trabalhanda ======
base = pd.read_csv('plano-saude2.csv')

# Variável X (as idades) que é independente e nela vai se fazer a previsão do custo do plano
x = base.iloc[:, 0:1].values

# Variável Y o valor do plano de saúde
y = base.iloc[:, 1].values

# ===== Importando e criando a árvore de decissão =====
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()

# ==== Gerando o aprendizado =====
regressor.fit(x, y)

# ===== Visualizando o score =====
score = regressor.score(x, y)

# Visualização do gráfico
import matplotlib.pyplot as plt
plt.scatter(x, y)

# ===== Realizando a regressão e visualizando no grafico =====
plt.plot(x, regressor.predict(x), color = 'red')
plt.title('Regressão com redes neurais')
plt.xlabel('Idade')
plt.ylabel('Custo')

# ===== Importação da biblioteca =====
import numpy as np
x_teste = np.arange(min(x), max(x), 0.1)
x_teste = x_teste.reshape(-1,1)
plt.scatter(x, y)
plt.plot(x_teste, regressor.predict(x_teste), color = 'red')
plt.title('Regressão com redes neurais')
plt.xlabel('Idade')
plt.ylabel('Custo')

# previsão do valor do plano de saúde para uma pessoa com 40 anos de idade
regressor.predict([[40]])