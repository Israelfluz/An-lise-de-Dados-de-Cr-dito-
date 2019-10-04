# =============================================================================

#            Objetivo é treinar o modelo de machine learn e depois fazer uma
#                          aplicação web com python e django
 
# =============================================================================

# Importação da biblioteca panda para manipulação, leitura e visualização de dados
import pandas as pd

# Utilizando o LabelEncoder para transformar as variáveis categorias em variáveis discretas                
from sklearn.preprocessing import LabelEncoder

# Importação do algoritmo naive bayes com GaussianNB e rodando o Navie Bayes para gerar a tabela de probabilidade 
from sklearn.naive_bayes import GaussianNB

# Comparativos de erros e acertos da base de dados
from sklearn.metrics import accuracy_score
import pickle

# Importação da base de dados que visa prêve qual é o risco de acidente
base = pd.read_csv('insurance.csv')

# Da base de dados serão utilizados os seguintes atributos
base.Age.unique() # Idade
base.RiskAversion.unique() # Risco do acidente
base.MakeModel.unique() # Tipo do carro
base.Accident.unique() # Chance de acidente


# Variável x com os atributos previsores
X = base.iloc[:, [2, 4, 9]].values

# Variável y é a classe que eu quero fazer a previsão
y = base.iloc[:, 8].values

# Transformando o atributo categórico para um atributo numérico com o labelencoder
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0])
X[:,1] = labelencoder.fit_transform(X[:,1])
X[:,2] = labelencoder.fit_transform(X[:,2])


# Criando o modelo do Naivebayes
modelo = GaussianNB()
modelo.fit(X, y) # treinando os dados com NaiveBayes

# Obtendo os resultados com a variável previsões
previsoes = modelo.predict(X)
accuracy_score(y, previsoes) # Avaliação do desempenho

pickle.dump(modelo, open('naivebayes_finalizado.sav', 'wb'))
