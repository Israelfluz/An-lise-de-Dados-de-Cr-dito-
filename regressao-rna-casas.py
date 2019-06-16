# ====== importação da biblioteca =======
import pandas as pd

# ===== carregando a base de dados a ser trabalhanda ======
base = pd.read_csv('house-prices.csv')

# ===== Variável X que são os atributos previssões ou independentes =====
X = base.iloc[:, 3:19].values

# ===== Variável Y que são os valores reais das casas onde vamos fazer a previsão =====
y = base.iloc[:, 2:3].values

# Importação e escalonamento da variável X
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)

# Escalonamento da variável de resposta Y 
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# ==== Criação da base de dadtos trienamento (70%) e base de teste (30%) =======
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)

# Importação e implemantação de rede neural
from sklearn.neural_network import MLPRegressor
regressor = MLPRegressor(hidden_layer_sizes = (9,9))

# ===== Gerando o treinamento =====
regressor.fit(X_treinamento, y_treinamento)

# ==== Visualizando o valor do score para saber como esta se comportando o previssor ====
score = regressor.score(X_treinamento, y_treinamento)

# ==== Visualizando o valor do score para saber como esta se comportando a resposta ====
regressor.score(X_teste, y_teste)

# ===== Realizando algumas previssões =====
previsoes = regressor.predict(X_teste)
y_teste = scaler_y.inverse_transform(y_teste)
previsoes = scaler_y.inverse_transform(previsoes)

# ===== Outra maneira de visualizar a diferença =====
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste, previsoes)



