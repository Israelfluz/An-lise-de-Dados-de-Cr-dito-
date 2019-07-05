# ===== Importação da biblioteca para visualização de graficos =====
import matplotlib.pyplot as plt

# ===== Importação da biblioteca para Manipulação, Leitura, Visualização de dados. =====
import pandas as pd

# ===== Importação do escalonador =====
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ===== Importação da biblioteca para visualização de graficos =====
import numpy as np

# ===== Carregando a base de dados =====
## ===== Obs: Essa base de dados é utilizada para fazermos classificação, mas vamos fazer agrupamento =====
base = pd.read_csv('credit-card-clients.csv', header = 1)

# ===== Variável que faz o somatório dos atributos que estão na base de dados =====
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6']

# ===== Variável X com a função i.loc com os indices da base de dados =====
X = base.iloc[:,[1,25]].values

# ===== Realizando o escalonamento para o agrupamento =====
scaler = StandardScaler()
X = scaler.fit_transform(X)


dbscan = DBSCAN(eps = 0.37, min_samples = 4)
previsoes = dbscan.fit_predict(X)

# ===== Na variável mostramos quantos grupos e na variável quantidade mostramos quantos elementos de cada grupo =====
unicos, quantidade = np.unique(previsoes, return_counts = True)

# ===== Agora visualizando os dados depois da definição de clusters e aprendizagem =====
plt.scatter(X[previsoes == 0, 0], X[previsoes == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[previsoes == 1, 0], X[previsoes == 1, 1], s = 100, c = 'orange', label = 'Cluster 2')
plt.scatter(X[previsoes == 2, 0], X[previsoes == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.xlabel('Limite')
plt.ylabel('Gastos')
plt.legend()

# ===== Realizando a concatenação ====== 
lista_clientes = np.column_stack((base, previsoes))

# ===== Ordenação dos clientes em seus grupos =====
lista_clientes = lista_clientes[lista_clientes[:,26].argsort()]