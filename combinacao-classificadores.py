# IMPORTAÇÃO DAS BIBLIOTECAS
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

# ====== CARREGANDO OS CLASSIFICADORES SALVOS ======
svm = pickle.load(open('svm_finalizado.sav', 'rb'))
random_forest = pickle.load(open('random_forest_finalizado.sav', 'rb'))
mlp = pickle.load(open('mlp_finalizado.sav', 'rb'))

# NOVO REGISTRO PARA FAZZER NOVA CLASSIFICAÇÃO
novo_registro = [[50000, 40, 5000]]

# TRANSFORMANDO O REGISTRO PARA UM NUMPY ARRAY
novo_registro = np.asarray(novo_registro)

# MODIFICANDO REGISTRISTRO DE ARRAY PARA LINHAS
novo_registro = novo_registro.reshape(-1, 1)

# =============== PADRONIZAÇÃO DOS VALORES ===============
scaler = StandardScaler()
novo_registro = scaler.fit_transform(novo_registro)

# ====== RETORNANDO O RESGISTRO PARA O FORMATO PADRÃO ========
novo_registro = novo_registro.reshape(-1, 3)

# ====== REALIZANDO UMA PREVISÃO DOS CLASSIFICADORES ======
resposta_svm = svm.predict(novo_registro)
resposta_random_forest = random_forest.predict(novo_registro)
resposta_mlp = mlp.predict(novo_registro)

# REALIZANDO A COMBINAÇÃO DA RESPOSTA DOS CLASSIFICADORES
paga = 0
nao_paga = 0

if resposta_svm[0] == 1:
    paga += 1
else:
    nao_paga += 1
    
if resposta_random_forest[0] == 1:
    paga += 1
else:
    nao_paga += 1
    
if resposta_mlp[0] == 1:
    paga += 1
else:
    nao_paga += 1
    
if paga > nao_paga:
    print('Cliente pagará o empréstimo')
elif paga == nao_paga:
    print('Resultado empatado')
else:
    print('Cliente não pagará o empréstimo')