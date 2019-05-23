#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 23:37:52 2019

@author: israelfaria
"""

import pandas as pd
base = pd.read_csv('credit-data.csv')
base.describe()
base.loc[base['age']< 0]

# Como tratar esses dados com informações problemáticas?

# 1º - apagar a coluna inteira (dependendo do caso é viável ou não)
base.drop('age', 1, inplace=True)

# 2º - apagar somente os registros com problema (os valores negativos)
base.drop(base[base.age < 0].index, inplace=True)

# 3º - preencher os dados manualmente('Possivelmente o mais correto e mais braçal, pois tem que entrar em contato com cada cliente')

# 4º - preenche ou calcular a idade média
base.mean() # Aqui vai aparecer a média geral de todos os dados desse arq.

base['age'].mean() # Aqui vai aparecer a média de uma coluna expecífica.
                   # Mas essa média inclui os valores negativos o que não é bom

base['age'][base.age > 0].mean() # Aqui colocando > 0 eu retiro os dados negativos

base.loc[base.age < 0, 'age']= 40.92 # Esse comando estabelece a média para os valores negativos que eu informo


# Tratando valores faltantes
pd.isnull(base['age']) # Mostra quando tem ou quando não tem valores nulos

base.loc[pd.isnull(base['age'])] 

classe = base.iloc[:, 4]

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3]= imputer.transform(previsores[:, 0:3])

# Escalonamento 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # A variável scaler foi criada para representar um objeto dessa classe
previsores = scaler.fit_transform(previsores)