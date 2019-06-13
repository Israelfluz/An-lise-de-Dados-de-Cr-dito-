# ====== importação da biblioteca =======
import pandas as pd

# ===== carregando a base de dados a ser trabalhanda ======
base = pd.read_csv('plano-saude2.csv')

# Variável X que é independente e nela vai se fazer a previsão do custo do plano
X = base.iloc[:, 0:1].values

y = base.iloc[:, 1].values

# Regressão linear simples
from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X, y)
score1 = regressor1.score(X, y)

# previsão do valor do plano de saúde para uma pessoa com 40 anos de idade
regressor1.predict([[40]])

# Visualização do gráfico
import matplotlib.pyplot as plt
plt.scatter(X, y)

# Visualizando a linha da regressão no grafico plotado
plt.plot(X, regressor1.predict(X), color = 'red')

# Criando um título para o grafico
plt.title('Regressão linear')

# Criando um título para o eixo X
plt.xlabel('Idade')

# Criando um título para o eixo Y
plt.ylabel('Custo')

# Regressão polinomial
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 4)
X_poly = poly.fit_transform(X)

regressor2 = LinearRegression()
regressor2.fit(X_poly, y)
score2 = regressor2.score(X_poly, y)

regressor2.predict(poly.transform([[40]]))

plt.scatter(X, y)
plt.plot(X, regressor2.predict(poly.fit_transform(X)), color = 'red')
plt.title('Regressão polinomial')
plt.xlabel('Idade')
plt.ylabel('Custo')