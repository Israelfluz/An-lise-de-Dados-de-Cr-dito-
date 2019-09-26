# ===== Importação da biblioteca para visualização de graficos =====
import matplotlib.pyplot as plt

# ===== Importação da biblioteca para computação científica =====
import numpy as np

# ===== Bilioteca Scipy para calculos científicos e estatística =====
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import AgglomerativeClustering

# ===== Importação do escalonador =====
from sklearn.preprocessing import StandardScaler

# ===== Varável X que contem a idade das pessoas =====
x=[20,  27,  21,  37,  46, 53, 55,  47,  52,  32,  39,  41,  39,  48,  48]  

# ===== Variável Y que contem os salários =====
y=[1000,1200,2900,1850,900,950,2000,2100,3000,5900,4100,5100,7000,5000,6500]

# ==== Visualização em forma de grafico das variáveis (x,y) =====
plt.scatter(x,y)

# ===== Criando a base de dados no formato Numpy array =====
base = np.array([[20,1000],[27,1200],[21,2900],[37,1850],[46,900],
                 [53,950],[55,2000],[47,2100],[52,3000],[32,5900],
                 [39,4100],[41,5100],[39,7000],[48,5000],[48,6500]])

# ===== Cirando o escalonamento da base de dados =====
scaler = StandardScaler()
base = scaler.fit_transform(base)

# ===== Cirando o dendrograma para saber quantos clusters serão definidos =====
dendrograma = dendrogram(linkage(base, method = 'ward'))
plt.title('Dendrograma')
plt.xlabel('Pessoas')
plt.ylabel('Distância Euclidiana')

# ===== Definindo o número de clusters =====
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')

# ===== Realizando o treinamento =====
previsoes = hc.fit_predict(base)

# ===== Agora visualizando os dados depois da definição de clusters e aprendizagem =====
plt.scatter(base[previsoes == 0, 0], base[previsoes == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(base[previsoes == 1, 0], base[previsoes == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(base[previsoes == 2, 0], base[previsoes == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.xlabel('Idade')
plt.ylabel('Salário')
plt.legend()

