# ====== importação da biblioteca =======
import pandas as pd

# ===== carregando a base de dados a ser trabalhanda ======
base = pd.read_csv('house-prices.csv')

# ===== Variável X que são os atributos previssões =====
X = base.iloc[:, 3:19].values

# ===== Variável Y que são os valores reais das casas =====
y = base.iloc[:, 2].values

# ==== Criação da base de dadtos trienamento (70%) e base de teste (30%) =======
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
# ===== Modelo de regressão polinominal =====
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()

# ===== Gerando o treinamento =====
regressor.fit(X_treinamento, y_treinamento)

# ==== Visualizando o valor do score para saber como esta se comportando o previssor ====
score = regressor.score(X_treinamento, y_treinamento)

# ===== Realizando algumas previssões =====
previsoes = regressor.predict(X_teste)

# ===== Outra maneira de visualizar a diferença =====
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste, previsoes)

regressor.score(X_teste, y_teste)

