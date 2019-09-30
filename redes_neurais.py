# Importando os datasets do sklearn
from sklearn import datasets

# Imporatando MultiLayer perceptron 
from sklearn.neural_network import MLPClassifier

# Fazendo a divisão da base de dados entre treinamento e teste
from sklearn.model_selection import train_test_split

# Comparativos de erros e acertos da base de dados
from sklearn.metrics import accuracy_score

# Visualizando a matrix de confusão
from yellowbrick.classifier import ConfusionMatrix

# Carregamentos do datasets
iris = datasets.load_iris()

# Dividindo a base de dados em treinamentp e teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(iris.data, iris.target,
                                                                  test_size = 0.3,
                                                                  random_state = 0)

modelo = MLPClassifier(verbose = True, hidden_layer_sizes=(5,4), max_iter = 10000)
modelo.fit(X_treinamento, y_treinamento)


previsoes = modelo.predict(X_teste)
accuracy_score(y_teste, previsoes)

# Matriz de confusão
confusao = ConfusionMatrix(modelo)
confusao.fit(X_treinamento, y_treinamento)
confusao.score(X_teste, y_teste)
confusao.poof()
