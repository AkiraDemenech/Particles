'''
Interface para treinamento e teste dos modelos 
'''
import streamlit 
import time
import back  
from matplotlib import pyplot
#from plotly import express

resultados = {}
mostra = {
	0: False # mostra somente o dataset atual
}


def principal (dataset_names = back.datasets, algorithms = back.classificadores, algorithm_args=back.argumentos):
	'''
		Função que será chamada para iniciar o processo.
		Configura e chama todas as demais, se ou quando necessárias.
	'''			
	streamlit.set_page_config(
		page_title='Busca por partículas exóticas'
	)
	
	dataset = streamlit.selectbox(
		'Escolha o dataset', 
		dataset_names, 
		on_change = lambda: print('Trocando de dataset')
	)				

	streamlit.write('Caso seu computador não seja capaz de rodar todo o dataset, reduza-o:')
	frac = streamlit.slider('Percentual do dataset para descartar', 0.0, 100.0, 70.0)
	random_state_frac = streamlit.number_input('Semente aleatória para a divisão do dataset', min_value=0, step=1)

	streamlit.divider()
	streamlit.header('Parâmetros do treinamento:')
	holdout = streamlit.slider('Percentual do dataset para hold-out', 0.01, 100.0, 20.0)

	random_state = streamlit.number_input('Semente aleatória para a divisão do hold-out', min_value=0, step=1)

	alg = streamlit.selectbox(
		'Escolha o algoritmo de classificação', 
		algorithms, 
		on_change = lambda: print('Trocando de algoritmo')
	)

	args = {}
	for k, descri in algorithm_args[alg]:
		tipo, validadora, inicial = algorithm_args[alg][k, descri]
		descri = f'{k}: {descri}'
		if tipo == int:
			v = streamlit.number_input(descri, value=inicial, min_value=0, step=1)
		elif tipo == float:	
			v = streamlit.number_input(descri, value=float(inicial), min_value=0.0, step=0.01)
		elif type(tipo) in (dict, list, tuple, set):	
			v = streamlit.selectbox(descri, tipo) # não tem valor inicial pré-definido
		args[k] = validadora(v)	

	training_state = streamlit.number_input('Semente aleatória para o treinamento', min_value=0, step=1)


	if streamlit.button(f'Carregar dataset {dataset} e treinar o modelo {alg}'):
		streamlit.divider()
		streamlit.header('Carregamento:')
		X,y = carregar_dataset(dataset, frac, random_state_frac)		 
		
		streamlit.header('Divisão:')
		X_train, X_test, y_train, y_test = dividir_dataset(X,y, holdout, random_state)
		print(y_train.value_counts())
		print(y_test.value_counts())
		X = y = None # descartar o dataset completo

		streamlit.write(len(X_train),'linhas de treino')
		streamlit.write(X_train.shape[1],'características')
		streamlit.write(X_train.head())		

	#	streamlit.write(y_train.value_counts())		

		possibilidades = ['Background', 'Evento de interesse']			
		freq = y_train.value_counts().reindex(range(len(possibilidades)), fill_value=0)

		fig, ax = pyplot.subplots() 
		ax.pie(freq, labels=possibilidades, autopct='%1.1f%%')		 
		fig.gca().add_artist(pyplot.Circle((0,0),0.7,fc='white')) # buraco da rosquinha no meio da torta/pizza
		pyplot.title('Frequências dos resultados no treinamento')
		streamlit.pyplot(fig)

		streamlit.write(len(X_test),'linhas de teste')
		streamlit.write(X_test.shape[1],'características')
		streamlit.write(X_test.head())	
		
	#	streamlit.write(y_test.value_counts())

		freq = y_test.value_counts().reindex(range(2), fill_value=0)
		
		fig, ax = pyplot.subplots() 
		ax.pie(freq, labels=possibilidades, autopct='%1.1f%%')		 
		fig.gca().add_artist(pyplot.Circle((0,0),0.7,fc='white')) # buraco da rosquinha no meio da torta/pizza
		pyplot.title('Frequências dos resultados nos testes')
		streamlit.pyplot(fig)	

		streamlit.header('Treinamento:')
		modelo = treinar_modelo(X_train, y_train, alg, args, training_state)
		X_train = y_train = None # descartar os dados de treinamento

		fig, ax = pyplot.subplots()
		importance = modelo.feature_importances_
		ind = back.numpy.argsort(importance)[::-1]
		print(importance, ind)

		pyplot.title(f'Importância das Características do Modelo {alg} Treinado')
		ax.bar(range(X_test.shape[1]), importance[ind], align='center')
		pyplot.xticks(range(X_test.shape[1]), back.numpy.array([str(i) for i in ind]), rotation=90)
		pyplot.xlim([-1, X_test.shape[1]])
		pyplot.ylabel('Importância')
		pyplot.xlabel('Característica')

		streamlit.pyplot(fig)
		
		streamlit.header('Teste:')
		metrics = testar_modelo(X_test, y_test, modelo)
		X_test = y_test = None # descartar os dados de teste

		if dataset not in resultados:
			resultados[dataset] = {}
		if alg not in resultados[dataset]:
			resultados[dataset][alg] = {a: [] for a in args}, {m: [] for m in metrics}
			mostra[alg] = False
		argumentos, medidas = resultados[dataset][alg]	
		for a in args:
			argumentos[a].append(args[a])
		for m in metrics:	
			medidas[m].append(metrics[m])

	streamlit.divider()
#	streamlit.write(resultados)

	for m in mostra:		
		def inverter (var=m):
			mostra[var] = not mostra[var]
		if m:
			streamlit.button(f'Mostrar somente os resultados de {m} para o dataset {dataset}' if mostra[m] else f'Mostrar todos os resultados para {m}', on_click=inverter)
		else:	
			streamlit.button(f'Mostrar somente os resultados para o dataset {dataset}' if mostra[m] else f'Mostrar os resultados de todos os datasets', on_click=inverter)
	 

	for d in resultados:		
		e = False

		for a in resultados[d]:				

			if d != dataset and (not mostra[0]) and (not mostra[a]):
				continue 

			streamlit.header(f'Resultados para {a} em {d}:')
			
			argu, mede = resultados[d][a]

			horizontal = streamlit.selectbox(
				f'Escolha a variável de x (hiperparâmetro de {a} em {d})', 
				argu
			)

		#	streamlit.write(argu[horizontal])			 
		#	valores = set(argu[horizontal])
			contagem = {}			
			tipos = [
				(str,),
				(int,float),
			#	(type(None),),	
				None
			]
			contatipo = {
				t: {}
				for t in tipos
			}
			for k in range(len(argu[horizontal])):
				v = argu[horizontal][k]
				if v in contagem:	
					contagem[v] += 1
				else:	
					contagem[v] = 1					

				for t in contatipo:	
					if v is t: # None
						v = str(v)
					elif t is None or type(v) not in t:	
						continue 

					if v not in contatipo[t]:
						contatipo[t][v] = {
							m: []
							for m in mede
						}	
					for m in mede:
						contatipo[t][v][m].append(mede[m][k])						

			xordem = {
				t: list(contatipo[t])
				for t in contatipo
			}			

			x = []
			y = {}

			for t in tipos:
				xordem[t].sort() # ordem crescente
				x.extend(xordem[t])

				for v in xordem[t]: 
					for m in contatipo[t][v]:
						y[v,m] = sum(contatipo[t][v][m]) / (len(contatipo[t][v][m]) + (len(contatipo[t][v][m]) == 0))


					
			valores = list(contagem)			
			

			fig, ax = pyplot.subplots() 
			ax.pie([contagem[v] for v in valores], labels=[str(v) for v in valores], autopct='%1.1f%%')		 
			fig.gca().add_artist(pyplot.Circle((0,0), 1-(1/len(valores)),fc='white')) # buraco da rosquinha no meio da torta/pizza
			pyplot.title(f'Frequências valores em {horizontal}')
			streamlit.pyplot(fig)
			
			fig, ax = pyplot.subplots()
			for m in mede:
				ax.plot([str(k) for k in x], [y[k,m] for k in x], 'o-', label=m)
			pyplot.title(f'Métricas (média aritmética) por {horizontal}')
			pyplot.ylabel('Métricas')
			pyplot.xlabel(horizontal)
			pyplot.legend()
			streamlit.pyplot(fig)

			def limpar (algoritmo=a):
				print('Removendo', algoritmo)
				r = resultados[d].pop(algoritmo)
				print(len(r), 'removidos.')

			if streamlit.button(f'Apagar histórico de {a} em {d}', on_click=limpar):
				print('Apagando',a,d) # não é permitido apagar diretamente aqui no IF pois isso mudaria o tamanho do dicionário durante a iteração (isso poderia ser resolvido puxando uma cópia do iterator, mas como são dois FOR seria uma gambiarra repetida)
				
			e = True 	

		if e and streamlit.button(f'Apagar todo o histórico de {d}', on_click=lambda t=d: resultados.pop(t)):		
			print('Apagando',d)
	
def carregar_dataset (dataset_name, desc_frac, desc_semente):
	status = streamlit.text('Carregando dataset....')

	print('\n', time.time())
	print('\n', 'Carregando', dataset_name)
	ti = time.time()				

	# chama função do back para carregar o dataset 
	X,y = back.carregar_dataset(dataset_name, desc_frac, desc_semente)

	tf = time.time()		

	print('\n', tf, ti, '\t', tf - ti)
	print('\n', 'Carregado', dataset_name)

	time.sleep(1 - (time.time() % 1))

	status.text('Dataset carregado!')
	streamlit.write('Tempo de carregamento:', tf - ti, 's')

	return X, y

def dividir_dataset (X, y, holdout_frac, holdout_semente):	

#	if desc_frac > 0:
	#	desc = streamlit.text(f'Removendo {desc_frac}% do dataset')
	#	X, X_descartado, y, y_descartado = back.dividir_dataset(X, y, desc_frac, desc_semente)
	#	desc.text(f'Utilizando {100-desc_frac}% do dataset')		
	div = streamlit.text(f'Separando {holdout_frac}% do dataset para treinamento')
	X_treinamento, X_teste, y_treinamento, y_teste = back.dividir_dataset(X, y, holdout_frac, holdout_semente)	
	div.text(f'Separados {holdout_frac}% do dataset para treinamento')

	return X_treinamento, X_teste, y_treinamento, y_teste

def treinar_modelo (X, y, tipo, argumentos, estado):
	status = streamlit.text(f'Treinando modelo {tipo}....')					
					
	print('\n', time.time())
	print('\n', 'Treinando', tipo)
	ti = time.time()				

	# chama função do back para carregar o dataset 
	modelo = back.treinar_modelo(X, y, tipo, argumentos, estado)

	tf = time.time()		

	print('\n', tf, ti, '\t', tf - ti)
	print('\n', 'Treinado')

	time.sleep(1 - (time.time() % 1))

	status.text(f'Modelo {tipo} treinado!')
	streamlit.write('Tempo de treinamento:', tf - ti, 's')

	return modelo

def testar_modelo (X, y, modelo):

	status = streamlit.text(f'Testando modelo {type(modelo).__name__}....')					
					
	print('\n', time.time())
	print('\n', 'Testando', type(modelo).__name__)
	ti = time.time()				

	# chama função do back para carregar o dataset 
	matrix, metrics = back.testar_modelo(X, y, modelo)

	tf = time.time()		

	print('\n', tf, ti, '\t', tf - ti)
	print('\n', 'Testado')

	time.sleep(1 - (time.time() % 1))

	status.text(f'Modelo {type(modelo).__name__} testado!')
	streamlit.write('Tempo de teste:', tf - ti, 's')	

	streamlit.write('Métricas de desempenho:')
	for m in metrics:
		streamlit.write(f'{m}: {metrics[m]}')

	streamlit.write('Matriz de confusão:')
	streamlit.write(matrix)	

	return metrics