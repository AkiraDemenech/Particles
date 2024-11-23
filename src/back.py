'''
Informações para utilização dos datasets e hiperparâmetros para cada modelo classificador 
'''
# py -m venv sklearn-env
# 	sklearn-env\Scripts\activate 
# 	source sklearn-env/bin/ctivate
# pip install -U scikit-learn 
# pip install pandas matplotlib
# pip install xgboost
# pip install streamlit
# streamlit run main.py

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

import pandas 
import numpy

def inalterado (x):
	return x

TOY = 'TOY (100 primeiras linhas de SUSY)'
YOT = 'YOT (100 últimas linhas de SUSY)'
BYEGGS = 'BYEGGS (1000 últimas linhas de HIGGS)'
SUSY = 'SUSY'
HIGGS = 'HIGGS'

datasets = {
	TOY:	'datasets/TOY.csv',
	YOT:	'datasets/YOT.csv',
	BYEGGS:	'datasets/BYEGGS.csv',
	SUSY:	'datasets/SUSY.csv',
	HIGGS:	'datasets/HIGGS.csv'
}

GINI = 'gini'
ENTROPY = 'entropy'
LOG_LOSS = 'log_loss'
criteria = [
	GINI, ENTROPY, LOG_LOSS
]

BEST = 'best'
RANDOM = 'random'
splitters = [
	BEST, RANDOM
]

DT = 'Árvore de Decisão'#'Decision tree'
D_ARGS = { # argumentos sobre profundidade
	('max_depth', 'Profundidade máxima das árvores (deixe 0 para ilimitada)'): (int, lambda p: p if p > 0 else None, 0)
}
T_ARGS = dict(D_ARGS)
T_ARGS.update({ # argumentos comuns às árvores 
	('criterion', 'Critério de escolha para divisão'): (criteria, inalterado, None),	
	('min_samples_split', 'Amostragem mínima para divisão'): (int, inalterado, 2),
	('min_samples_leaf', 'Amostragem mínima por folha'): (int, inalterado, 1)
})

DT_ARGS = dict(T_ARGS)
DT_ARGS.update({
	('splitter', 'Escolha de separação'): (splitters, inalterado, None)
})

EN_ARGS = { # argumentos comuns aos ensambles
	('n_estimators', 'Número de árvores'): (int, inalterado, 10)	
}

RF = 'Random forest'
RF_ARGS = dict(T_ARGS)
RF_ARGS.update(EN_ARGS)

ADB = 'Adaptive Boosting'
ADB_ARGS = dict(EN_ARGS)
ADB_ARGS.update({
	('learning_rate', 'Taxa de aprendizado'): (float, inalterado, 1)
})

XGB = 'XGradient Boosting'
XGB_ARGS = dict(EN_ARGS)
XGB_ARGS.update(ADB_ARGS)
XGB_ARGS.update(D_ARGS)

classificadores = {
	DT: DecisionTreeClassifier,
	RF: RandomForestClassifier,
	ADB: AdaBoostClassifier,
	XGB: XGBClassifier
}

argumentos = {
	DT: DT_ARGS,
	RF: RF_ARGS,
	ADB: ADB_ARGS,
	XGB: XGB_ARGS
}

# Funções para utilizar os modelos

def carregar_dataset (dataset, remove, seed, arquivos = datasets):

	arquivo = arquivos[dataset]

	ignore = None
	if remove > 0:
		
		print('Contando linhas....')
		numpy.random.seed(seed)
		qtd = sum(True for ln in open(arquivo, 'r'))

		remove_abs = (qtd * remove / 100).__ceil__()	
		print(f'Ignorando {remove}% ({qtd})')
		ignore = numpy.random.choice(range(qtd),remove_abs, replace=False)
		
	
	print('Iniciando carregamento propriamente dito')
	dados = pandas.read_csv(arquivo, skiprows=ignore, header=None)#, iterator=True, chunksize=1000)
	print(dados.head())

	X = dados.drop(columns=[0]) # todas as colunas, exceto a primeira
	y = dados[0] # primeira coluna é a classificação

	return X,y

def dividir_dataset (X, y, porcentual_teste, seed):	

	return train_test_split(X, y, test_size=porcentual_teste/100, random_state=seed)

def treinar_modelo (X, y, alg, hiperpar, seed, algoritmos = classificadores):
					
	mod = algoritmos[alg](random_state=seed, **hiperpar)
	mod.fit(X, y)

	return mod

def testar_modelo (X, y, modelo):	
	y_pred = modelo.predict(X)

	confus = confusion_matrix(y, y_pred)

	print(confus)
	print(classification_report(y, y_pred))

	return confus, {
		'Acurácia (accuracy)': accuracy_score(y, y_pred),
		'Precisão (precision)': precision_score(y, y_pred),
		'Revocação (recall)': recall_score(y, y_pred),
		'F1': f1_score(y, y_pred)
	}
	
