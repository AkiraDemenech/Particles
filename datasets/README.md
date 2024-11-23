Além dos dois datasets publicados por Daniel Whiteson em 2014, também foram extraídas versões simplificadas para o desenvolvimento do projeto e sua demonstração em ambientes com pouca memória ou tempo limitado.

[HIGGS.csv](./HIGGS.csv):
produção de Bóson de Higgs

* [BYEGGS.csv](./BYEGGS.csv):
recorte das 1000 últimas linhas para testes rápidos 

[SUSY.csv](./SUSY.csv):
produção de partículas supersimétricas

* [TOY.csv](./TOY.csv):
recorte das 100 primeiras linhas para debug 

* [YOT.csv](./YOT.csv):
recorte das 100 últimas linhas para debug 

Os datasets originais são muito grandes para o GitHub, então você precisa reconstituí-los.
Para os arquivos compactados caberem, foram divididos em vários pequenos.
Para descompactá-los, utilize:

	unzip HIGGS.parte.zip
	unzip SUSY.parte.zip

Mantendo os arquivos HIGGS.parte.z* e SUSY.parte.z* neste mesmo diretório.