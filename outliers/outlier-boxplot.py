# ===== Importação da biblioteca para Manipulação, Leitura, Visualização de dados. =====
import pandas as pd

# ===== Carregando a base de dados =====
base = pd.read_csv('credit-data.csv')

# ===== Apagando dados não preenchidos os NAN =====
base = base.dropna()

# outliers idade
# ===== Importação da biblioteca para visualização de graficos =====
import matplotlib.pyplot as plt
plt.boxplot(base.iloc[:,2], showfliers = True)
# ===== Capturando os outliers =====
outliers_age = base[(base.age < -20)]

# outliers loan (loan contem a dívida)
plt.boxplot(base.iloc[:,3])
outliers_loan = base[(base.loan > 13400)]