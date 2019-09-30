# Importando os datasets disponíveis da biblioteca sklearn
from sklearn import datasets

# Importação da classe KNeighborsClassifier do pacote neighbors do sklearn
from sklearn.neighbors import KNeighborsClassifier

# Carregamento da base de dados Iris que classifica tipos de plantas de acordo com suas características
iris = datasets.load_iris()


# Selecionando uma das plantas para fazer o cálculo da distância
iris_teste = iris.data[0,:]

# Buscando a classe (target) correta da planta 
iris_teste_classe = iris.target[0]

# Variável que contem os atributos previsores que são os dados para o treinamento
X = iris.data[1:150,:]

# Variável que contem as classes
y = iris.target[1:150]


# Variável Knn que recebe o KNeighborsClassifier que fará o cálculo de singularidade (os vizinhos mais próximos)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X, y) # Aqui acontece o treinamento 


previsao = knn.predict(iris_teste.reshape(1,-1))
