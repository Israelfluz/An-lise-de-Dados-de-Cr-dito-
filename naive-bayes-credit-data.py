import pandas as pd

# Carregadno a base de dados
base = pd.read_csv('credit-data.csv')

# Corrigindo valores com idades negativas
base.loc[base.age < 0, 'age'] = 40.92
               
# Fazendo a divisão entre previsores e classes
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

# classe Imputer para tratar valores não informados ou desconhecidos
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

# Scaler transforma todos os valores numérios em uma escala
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# Divisão da base de treinamento e teste
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25, random_state=0)

# Rodando o Navie Bayes para gerar a tabela de probabilidade 
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

# Comparativos de erros e acertos da base de dados
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)