# ===== Importação da biblioteca para Manipulação, Leitura, Visualização de dados. =====
import pandas as pd

# ===== Carregando a base de dados =====
base = pd.read_csv('census.csv')

# ===== Realizando a divisão dos previsores =====
previsores = base.iloc[:, 0:14].values
# ===== Variável classe =====
classe = base.iloc[:, 14].values
                
# ===== Obs: O dados constantes em previsores e classe estão no formato categórico ======
# ===== Transformando os dados categóricos com o LabelEncoder =====               
from sklearn.preprocessing import LabelEncoder
labelencoder_previsores = LabelEncoder()
previsores[:, 1] = labelencoder_previsores.fit_transform(previsores[:, 1])
previsores[:, 3] = labelencoder_previsores.fit_transform(previsores[:, 3])
previsores[:, 5] = labelencoder_previsores.fit_transform(previsores[:, 5])
previsores[:, 6] = labelencoder_previsores.fit_transform(previsores[:, 6])
previsores[:, 7] = labelencoder_previsores.fit_transform(previsores[:, 7])
previsores[:, 8] = labelencoder_previsores.fit_transform(previsores[:, 8])
previsores[:, 9] = labelencoder_previsores.fit_transform(previsores[:, 9])
previsores[:, 13] = labelencoder_previsores.fit_transform(previsores[:, 13])

# ====== Escalonamento para deixar os valores todos na mesma escala =====
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)

# ===== Divisão da base de dados =====
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

# ===== Redução de dimensionalidade e importação do KernelPCA =====
from sklearn.decomposition import KernelPCA

# ===== Transformando os 14 atributos na quantidade desejada =====
kpca = KernelPCA(n_components = 6, kernel = 'rbf')
previsores_treinamento = kpca.fit_transform(previsores_treinamento)
previsores_teste = kpca.transform(previsores_teste)

from sklearn.ensemble import RandomForestClassifier
classificador = RandomForestClassifier(n_estimators = 40, criterion = 'entropy', random_state = 0)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)