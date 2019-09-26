# ====== importação da biblioteca =======
import pandas as pd

# ===== carregando a base de dados a ser trabalhanda ======
base = pd.read_csv('house-prices.csv')

# ===== Variável X que são os atributos previssões ou independentes =====
X = base.iloc[:, 3:19].values

# ===== Variável Y que são os valores reais das casas onde vamos fazer a previsão =====
y = base.iloc[:, 2:3].values

# Importando e implementando o escalonamento
from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# ==== Criação da base de dadtos trienamento (70%) e base de teste (30%) =======
from sklearn.model_selection import train_test_split
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
# Importação e implementação do algoritmo SVR
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')

# ===== Gerando o treinamento =====
regressor.fit(X_treinamento, y_treinamento)
score = regressor.score(X_treinamento, y_treinamento)

# Score feito na base de dados teste
regressor.score(X_teste, y_teste)

previsoes = regressor.predict(X_teste)
y_teste = scaler_y.inverse_transform(y_teste)
previsoes = scaler_y.inverse_transform(previsoes)

# ===== Outra maneira de visualizar a diferença =====
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste, previsoes)
