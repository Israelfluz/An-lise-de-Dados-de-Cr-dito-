# ===== Importação da biblioteca para Manipulação, Leitura, Visualização de dados. =====
import pandas as pd

# ===== Carregando a base de dados =====
base = pd.read_csv('credit-data.csv')
# ===== Apagando dados não preenchidos os NAN =====
base = base.dropna()

from pyod.models.knn import KNN # kNN detector
detector = KNN()
# ==== Treinamento =====
detector.fit(base.iloc[:,1:4])

# ===== Detectando os outliers
previsoes = detector.labels_
# ==== Avaliando a confiança =====
confianca_previsoes = detector.decision_scores_

# ===== Fazendo uma lista dos outliers =====
outliers = []
for i in range(len(previsoes)):
    #print(previsoes[i])
    if previsoes[i] == 1:
        outliers.append(i)
        
lista_outliers = base.iloc[outliers, :]
    