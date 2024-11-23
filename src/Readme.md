[back.py](./back.py) possui todos os dados estruturados para utilizar os datasets e o formato genérico de utilização dos modelos (definindo os hiperparâmetros, seus tipos e valores padrão).

[front.py](./front.py) cria a interface no Streamlit para permitir controle direto do usuário sobre os modelos, de acordo com a organização de [back.py](./back.py).

[main.py](./main.py) simplesmente chama [front.py](./front.py).