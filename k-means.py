# Importando os datasets disponíveis da biblioteca sklearn
from sklearn import datasets

# Importação da matriz de confusão 
from sklearn.metrics import confusion_matrix

# Importação da bibblioteca matplotlib com recursos para a geração de gráficos 2D 
import matplotlib.pyplot as plt

# Importando o KMeans do pacote cluster no sklearn
from sklearn.cluster import KMeans


# Carregamento da base de dados Iris que classifica tipos de plantas de acordo com suas características
iris = datasets.load_iris()


# Variável clusters para gerar e indicar o número de clusters eu quero para o agrupamento
cluster = KMeans(n_clusters = 3)
cluster.fit(iris.data) # Agrupando com o método fit

# Verificando em qual cluster cada registro está com a variável previsões
previsoes = cluster.labels_


# Visualizando os centroides
centroides = cluster.cluster_centers_

# Visualizando os resultados
resultados = confusion_matrix(iris.target, previsoes)


# Visualizando o gráfico

plt.scatter(iris.data[previsoes == 0, 0], iris.data[previsoes == 0, 3], 
            c = 'green', label = 'Setosa')

plt.scatter(iris.data[previsoes == 1, 0], iris.data[previsoes == 1, 3], 
            c = 'red', label = 'Versicolor')

plt.scatter(iris.data[previsoes == 2, 0], iris.data[previsoes == 2, 3], 
            c = 'blue', label = 'Virgica')
plt.legend()