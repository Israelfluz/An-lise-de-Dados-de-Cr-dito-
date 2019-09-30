# Importação da biblioteca panda para manipulação, leitura e visualização de dados
import pandas as pd

# Fazendo a divisão da base de dados entre treinamento e teste
from sklearn.model_selection import train_test_split

# Importação do algoritmo naive bayes com GaussianNB e rodando o Navie Bayes para gerar a tabela de probabilidade 
from sklearn.naive_bayes import GaussianNB

# Importação accuracy_score
from sklearn.metrics import accuracy_score

# Importação feature_selection no sklearn que com o chi2 fará a seleção dos melhores atributos
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

# Carregando a base de dados para seleção de atributos
anuncio = pd.read_csv('ad.data', header = None)
anuncio[1558].unique()

X = anuncio.iloc[:,0:1558].values
y = anuncio.iloc[:, 1558].values


# Dividindo a base de dados em treinamento e teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)


# Modelo com todos os atributos
modelo1 = GaussianNB()
modelo1.fit(X_treinamento, y_treinamento)
previsoes1 = modelo1.predict(X_teste)
accuracy_score(y_teste, previsoes1)

# Seleção de atributos
selecao = SelectKBest(chi2, k=7)
X_novo = selecao.fit_transform(X, y)

# Colunas selecionadas
colunas = selecao.get_support()

X_treinamento_novo, X_teste_novo, y_treinamento, y_teste = train_test_split(X_novo, y,
                                                                  test_size = 0.3,
                                                                  random_state = 0)
# Modelo com seleção de atributos
modelo2 = GaussianNB()
modelo2.fit(X_treinamento_novo, y_treinamento)
previsoes2 = modelo2.predict(X_teste_novo)
accuracy_score(y_teste, previsoes2)


