# Importação da bibblioteca matplotlib com recursos para a geraÃ§Ã£o de grÃ¡ficos 2D 
import matplotlib.pyplot as plt

# Imporatação do Keras para a classe Sequential
from keras.models import Sequential

# Importação do keras.layers para a camada densa e dropout
from keras.layers import Dense, Dropout

# Importação do keras.utils
from keras.utils import np_utils

# Importação da biblioteca para computação científica
import numpy as np

# Importação da matriz de confusão
from sklearn.metrics import confusion_matrix

# Importação do datasets e base de dados mnist
from keras.datasets import mnist


(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()

# Dividindo a base de dados em treinamentp e teste


# As imagens da base de dados
plt.figure()

plt.subplot(2, 2, 1)
plt.imshow(X_treinamento[0], cmap = 'gray')
plt.title(y_treinamento[0])

plt.subplot(2, 2, 2)
plt.imshow(X_treinamento[1], cmap = 'gray')
plt.title(y_treinamento[1])

plt.subplot(2, 2, 3)
plt.imshow(X_treinamento[3], cmap = 'gray')
plt.title(y_treinamento[3])

plt.subplot(2, 2, 4)
plt.imshow(X_treinamento[4], cmap = 'gray')
plt.title(y_treinamento[4])

# Pré-processamento
X_treinamento = X_treinamento.reshape((len(X_treinamento), np.prod(X_treinamento.shape[1:])))
X_teste = X_teste.reshape((len(X_teste), np.prod(X_teste.shape[1:])))

X_treinamento = X_treinamento.astype('float32')
X_teste = X_teste.astype('float32')


# Normalizando para que o processamento fique mais rápido
X_treinamento /= 255
X_teste /= 255

# Realizando uma transformação com np_utils
y_treinamento = np_utils.to_categorical(y_treinamento, 10)
y_teste = np_utils.to_categorical(y_teste, 10)


# Construção do modelo da rede neural
# 784 - 64 - 64 - 64 - 10
modelo = Sequential()
modelo.add(Dense(units = 64, activation = 'relu', input_dim = 784)) # Primeira camada (entrada) do tipo densa
modelo.add(Dropout(0.2)) # A primeira camada recebe uma camada com dropout que zera uma porção do neurônios para evitar overfiting
modelo.add(Dense(units = 64, activation = 'relu')) # Segunda camada densa
modelo.add(Dropout(0.2)) # A segunda camada recebe uma camada com dropout que zera uma porção do neurônios para evitar overfiting
modelo.add(Dense(units = 64, activation = 'relu')) # Terceira camada
modelo.add(Dropout(0.2)) # A terceira camada recebe uma camada com dropout que zera uma porção do neurônios para evitar overfiting
modelo.add(Dense(units = 10, activation = 'softmax')) # Camada de saída

# Sumário do modelo da rede neural
modelo.summary()

# fazendo o ajuste dos pesos
modelo.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
               metrics = ['accuracy'])

# Efetivando o treinamento
historico = modelo.fit(X_treinamento, y_treinamento, epochs = 20,
                       validation_data = (X_teste, y_teste))


# Visualizando um gráfico dos dados
historico.history.keys()
plt.plot(historico.history['val_loss'])
plt.plot(historico.history['val_acc'])

# Gerando uma matriz de confusão
previsoes = modelo.predict(X_teste)
y_teste_matriz = [np.argmax(t) for t in y_teste]
y_previsoes_matriz = [np.argmax(t) for t in previsoes]
confusao = confusion_matrix(y_teste_matriz, y_previsoes_matriz) # Comparativo entre y_previsões e y_teste_matriz)

# Fazendo uma nova classificação
y_treinamento[20]
novo = X_treinamento[20]
novo = np.expand_dims(novo, axis = 0) # Colocando no formato de linha
pred = modelo.predict(novo)

