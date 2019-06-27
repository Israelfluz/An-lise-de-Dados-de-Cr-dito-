# ====== importação das bibliotecas =======
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

# ====== Criando as variáveis X e Y ======
x, y = make_blobs(n_samples = 200, centers = 5)
plt.scatter(x[:,0], x[:,1])

# ===== Aplicando o algoritmo K-means =====
kmeans = KMeans(n_clusters = 5)
kmeans.fit(x)

# ==== Observando se os dados estão sendo colocados no lugar certo ====
previsoes = kmeans.predict(x)
plt.scatter(x[:,0], x[:,1], c = previsoes)
