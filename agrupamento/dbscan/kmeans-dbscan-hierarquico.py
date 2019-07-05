# ===== Importação da biblioteca para visualização de graficos =====
import matplotlib.pyplot as plt

# ===== Importação do algoritmo KMeans =====
from sklearn.cluster import KMeans

# ===== Importação do algoritmo de clusterização hierárquica =====
from sklearn.cluster import AgglomerativeClustering

# ===== Importação do algoritmo DBSCAN =====
from sklearn.cluster import DBSCAN

# ===== Importação de datasets =====
from sklearn import datasets

# ===== Importação da biblioteca para visualização de graficos =====
import numpy as np

# ==== Criando as variáveis X e Y no datasets  de forma aleatoria =====
x, y = datasets.make_moons(n_samples = 1500, noise = 0.09)
# ===== Visualização dos dados no grafico resultante do make_mooons ======
plt.scatter(x[:, 0], x[:, 1], s = 5)

# ==== Criação dos vetores e suas cores para se ver no gráfico =====
cores = np.array(['red', 'blue'])

# ===== Comparativo com o algoritmo KMeans ======
# ===== Criando a variável Kmeans para separar os dados e criar os clusters ====
kmeans = KMeans(n_clusters = 2)

# ===== Criando as previsões e realizando o treinamento =====
previsoes = kmeans.fit_predict(x)

# ==== Visualizando o resultados das variáveis no grafico
plt.scatter(x[:, 0], x[:, 1], s = 5, color = cores[previsoes])

# ===== Comparativo com o algoritmo hierárquico e cluster =====
# ===== Definindo o número de clusters =====
hc = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')

# ===== Realizando o treinamento =====
previsoes = hc.fit_predict(x)
# ==== Visualizando o resultados das variáveis no grafico
plt.scatter(x[:, 0], x[:, 1], s = 5, color = cores[previsoes])

# ==== Comparativo com o algoritmo DBSCAN =====
dbscan = DBSCAN(eps = 0.1)
# ===== Realizando o treinamento =====
previsoes = dbscan.fit_predict(x)
# ==== Visualizando o resultados das variáveis no grafico
plt.scatter(x[:, 0], x[:, 1], s = 5, color = cores[previsoes])