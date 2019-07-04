# ===== Importação da biblioteca para visualização de graficos =====
import matplotlib.pyplot as plt

# ===== Importação da biblioteca para Manipulação, Leitura, Visualização de dados. =====
import pandas as pd

# ===== Bilioteca Scipy para calculos científicos e estatística =====
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# ===== Importação do escalonador =====
from sklearn.preprocessing import StandardScaler

# ===== Importação da biblioteca para computação científica =====
import numpy as np

# ===== Carregando a base de dados =====
base = pd.read_csv('credit-card-clients.csv', header = 1)

# ===== Variável que faz o somatório dos atributos que estão na base de dados =====
base['BILL_TOTAL'] = base['BILL_AMT1'] + base['BILL_AMT2'] + base['BILL_AMT3'] + base['BILL_AMT4'] + base['BILL_AMT5'] + base['BILL_AMT6']

# ===== Variável X com a função i.loc com os indices da base de dados =====
X = base.iloc[:,[1,25]].values

# ===== Realizando o escalonamento para o agrupamento =====
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ===== Cirando o dendrograma para saber quantos clusters serão definidos =====
dendrograma = dendrogram(linkage(X, method = 'ward'))

# ===== Definindo o número de clusters =====
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')

# ===== Realizando o treinamento =====
previsoes = hc.fit_predict(X)

# ===== Agora visualizando os dados depois da definição de clusters e aprendizagem =====
plt.scatter(X[previsoes == 0, 0], X[previsoes == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[previsoes == 1, 0], X[previsoes == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[previsoes == 2, 0], X[previsoes == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.xlabel('Limite')
plt.ylabel('Gastos')
plt.legend()
