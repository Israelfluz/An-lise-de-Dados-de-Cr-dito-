# ===== Importação da biblioteca para computação científica =====
import numpy as np

# ===== Importação da biblioteca para visualização de graficos =====
import matplotlib.pyplot as plt

# ===== Importação do escalonador =====
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

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

# ===== Gerando o aprendizado =====
dbscan = DBSCAN(eps = 0.95, min_samples = 2)
dbscan.fit(base)
previsoes = dbscan.labels_

# ===== Criando vetores =====
cores = ["g.", "r.", "b."]
for i in range(len(base)):
    plt.plot(base[i][0], base[i][1], cores[previsoes[i]], markersize = 15)
