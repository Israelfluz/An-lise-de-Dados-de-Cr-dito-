#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:26:00 2019

@author: israelfaria
"""
import pandas as pd

base = pd.read_csv('census.csv')

# Transformando variáveis categóricas
# Isso porque temos muitas variáveis como strings 
# E elas tem que ser transformadas em numéricas
previsores = base.iloc[:, 0:14]. values
classe = base.iloc[:, 14].values

# LabelEnconder é responsável pela transformação
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
labelencoder_previsores = LabelEncoder() # Criando o objeto labalencoder_previsores que recebe 

# O que esse código abaixo faz é ver a transformação esperada na base.
# que é transformar uma variável string em numérica 
# Neste caso o parametro 1 é coluna workclass
# Mas tem que fazer para todos os outros que são string
#labels = labelencoder_previsores.fit_transform(previsores[:, 1]) 

previsores[:, 1] = labels = labelencoder_previsores.fit_transform(previsores[:, 1]) 
previsores[:, 3] = labels = labelencoder_previsores.fit_transform(previsores[:, 3]) 
previsores[:, 5] = labels = labelencoder_previsores.fit_transform(previsores[:, 5]) 
previsores[:, 6] = labels = labelencoder_previsores.fit_transform(previsores[:, 6]) 
previsores[:, 7] = labels = labelencoder_previsores.fit_transform(previsores[:, 7]) 
previsores[:, 8] = labels = labelencoder_previsores.fit_transform(previsores[:, 8]) 
previsores[:, 9] = labels = labelencoder_previsores.fit_transform(previsores[:, 9]) 
previsores[:, 13] = labels = labelencoder_previsores.fit_transform(previsores[:, 12]) 

# Variável 'Dummpy' O "OneHotEncoder" faz com que um valor não venha ser mais importante que outro
onehotencoder = OneHotEncoder(categorical_features=[1, 3, 5, 6, 7, 8, 9, 13])
previsores = onehotencoder.fit_transform(previsores).toarray()

# Transformando a classe
labelencoder_classe = LabelEncoder()
classe = labelencoder_classe.fit_transform(classe)

# Importação do Scaler para escalonamento
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
