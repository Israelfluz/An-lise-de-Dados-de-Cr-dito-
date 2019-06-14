# ====== importação da biblioteca =======
import pandas as pd

# ===== carregando a base de dados a ser trabalhanda ======
base = pd.read_csv('plano-saude2.csv')

# ===== Variável X que são os atributos previssões as idades =====
X = base.iloc[:, 0:1].values

# Variável Y o valor do plano de saúde
y = base.iloc[:, 1].values

# ===== Importando e criando ou implemantando o Random Forest =====
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10)

# ==== Gerando o aprendizado =====
regressor.fit(X, y)

# ===== Visualizando o score =====
score = regressor.score(X, y)

# ===== Importação da biblioteca =====
import numpy as np
X_teste = np.arange(min(X), max(X), 0.1)
X_teste = X_teste.reshape(-1,1)

# Visualização do gráfico
import matplotlib.pyplot as plt
plt.scatter(X, y)

# ===== Realizando a regressão e visualizando no grafico =====
plt.plot(X_teste, regressor.predict(X_teste), color = 'red')
plt.title('Regressão com random forest')
plt.xlabel('Idade')
plt.ylabel('Custo')

# previsão do valor do plano de saúde para uma pessoa com 40 anos de idade
regressor.predict([[40]])