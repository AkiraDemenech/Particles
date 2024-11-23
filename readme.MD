# Busca de partículas exóticas com árvores

Utilizando os dados do artigo: 

> Baldi, P., Sadowski, P., and Whiteson, D. (2014). 
Searching for exotic particles in high-energy physics with deep learning. 
_Nature Communications_, 5(1).
DOI: [10.1038/ncomms5308](https://doi.org/10.1038/ncomms5308)

* [HIGGS](https://archive.ics.uci.edu/dataset/280/higgs):
classificação de processo que produz o Bóson de Higgs contra um ruído de fundo

* [SUSY](https://archive.ics.uci.edu/dataset/279/susy):
classificação de processo que produz partículas supersimétricas contra um ruído de fundo

* [HEPMASS](https://archive.ics.uci.edu/dataset/347/hepmass):
classificação de processo que produz uma nova partícula de massa desconhecida contra um ruído de fundo

## Instalação

[Instale o scikit-learn](https://scikit-learn.org/stable/install.html) na máquina.

Após isso, é fortemente recomendada a execução em um ambiente virtual para melhor gerenciar dependências.

### Ambiente virtual em Linux:

	python3 -m venv sklearn-env
	source sklearn-env/bin/activate
	pip3 install -U scikit-learn
	pip3 install pandas streamlit xgboost matplotlib   
	
### Ambiente virtual em Windows:

	python -m venv sklearn-env	
	sklearn-env\Scripts\activate.bat
	pip install -U scikit-learn
	pip install pandas streamlit xgboost matplotlib
	
## Execução

A partir do diretório raíz, execute o script principal no [Streamlit](https://streamlit.io/)

	streamlit run src/main.py
	
## Documentação 
 
* [Vídeo](https://drive.google.com/file/d/1KaD92wdk-Tcg5ttq0VPN5ZVDYNHxf-jq/view)

* [Paper](doc/particles.pdf)