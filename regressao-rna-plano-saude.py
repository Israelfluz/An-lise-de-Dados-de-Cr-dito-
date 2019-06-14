# ====== importação da biblioteca =======
import pandas as pd
import numpy as np

# ===== carregando a base de dados a ser trabalhanda ======
base = pd.read_csv('plano-saude2.csv')

# ===== Variável X que são os atributos previssões as idades =====
X = base.iloc[:, 0:1].values

# Variável Y o valor do plano de saúde
y = base.iloc[:, 1:2].values

# Importação e implementação do escalonamento
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# Importação e implemantação de rede neural
from sklearn.neural_network import MLPRegressor

# Aplicando a regressão
regressor = MLPRegressor()

# Gerando o aprendizado
regressor.fit(X, y)

regressor.score(X, y)

# Visualização do gráfico com a regressão
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color = 'red')
plt.title('Regressão com redes neurais')
plt.xlabel('Idade')
plt.ylabel('Custo')

# Previsão o valor do plano de saúde para uma pessoa que tem 40 anos.
previsao = scaler_y.inverse_transform(regressor.predict(scaler_x.transform(np.array(40).reshape(1, -1))))