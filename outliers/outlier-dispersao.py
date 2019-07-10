# ===== Importação da biblioteca para Manipulação, Leitura, Visualização de dados. =====
import pandas as pd

# ===== Carregando a base de dados =====
base = pd.read_csv('credit-data.csv')

# ===== Apagando dados não preenchidos os NAN =====
base = base.dropna()
base.loc[base.age < 0, 'age'] = 40.92

# income x age
# ===== Importação da biblioteca para visualização de graficos =====
import matplotlib.pyplot as plt
plt.scatter(base.iloc[:,1], base.iloc[:,2])

# income x loan
plt.scatter(base.iloc[:,1], base.iloc[:,3])

# age x loan
plt.scatter(base.iloc[:,2], base.iloc[:,3])

# ===== Carregando a base de dados =====
base_census = pd.read_csv('census.csv')

# age x final weight
plt.scatter(base_census.iloc[:, 0], base_census.iloc[:,2])