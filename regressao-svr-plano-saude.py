# ====== importação da biblioteca =======
import pandas as pd
import numpy as np 

# ===== carregando a base de dados a ser trabalhanda ======
base = pd.read_csv('plano-saude2.csv')

# ===== Variável X que são os atributos previssões as idades =====
X = base.iloc[:, 0:1].values

# Variável Y o valor do plano de saúde
y = base.iloc[:, 1:2].values

# Importando e implementando kernel linear 
from sklearn.svm import SVR
regressor_linear = SVR(kernel = 'linear')

# ==== Gerando o aprendizado =====
regressor_linear.fit(X, y.ravel())

# Visualização do gráfico
import matplotlib.pyplot as plt
plt.scatter(X, y)

# ===== Realizando a regressão e visualizando no grafico =====
plt.plot(X, regressor_linear.predict(X), color = 'red')
regressor_linear.score(X, y)

# # Importando e implementando kernel poly
regressor_poly = SVR(kernel = 'poly', degree = 3)

# ==== Gerando o aprendizado =====
regressor_poly.fit(X, y.ravel())

plt.scatter(X, y)

# ===== Realizando a regressão e visualizando no grafico =====
plt.plot(X, regressor_poly.predict(X), color = 'red')
regressor_poly.score(X, y)

# Importando e implementando kernel rbf (Esse kernel é o mais utilizado)
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()

# ==== Gerando o aprendizado =====
X = scaler_x.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

regressor_rbf = SVR(kernel = 'rbf')
regressor_rbf.fit(X, y)

plt.scatter(X, y)
plt.plot(X, regressor_rbf.predict(X), color = 'red')
regressor_rbf.score(X, y)

previsao1 = scaler_y.inverse_transform(regressor_linear.predict(scaler_x.transform(np.array(40).reshape(-1,1))))
previsao2 = scaler_y.inverse_transform(regressor_linear.predict(scaler_x.transform(np.array(40).reshape(-1,1))))
previsao3 = scaler_y.inverse_transform(regressor_linear.predict(scaler_x.transform(np.array(40).reshape(-1,1))))
