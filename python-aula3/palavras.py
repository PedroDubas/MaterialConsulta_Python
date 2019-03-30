#!/usr/bin/env python3
"""Recupera e Imprime as palavras de um URL.

Uso:
	python3 palavras.py <URL>
"""

import sys
from urllib.request import urlopen

def recuperar_palavras(url):
	"""Recupera uma lista de palavras de uma URL.
	Args:
		url: A URL de um documento no formato UTF-8.
	Returns:
		Uma lista de strings contendo as palavras do documento.
	"""
	with urlopen(url) as musica:
		palavras_musica = []
		for linha in musica:
			palavras_linha = linha.decode('utf-8').split()
			for palavra in palavras_linha:
				palavras_musica.append(palavra)
	return palavras_musica


def imprimir_items(items):
	"""Imprime um item por linha.
	Args:
		Uma serie iterável de items para imprimir.
	"""
	for item in items:
		print(item)


def main(url):
	"""Imprime cada palavra de um documento de texto de uma URL.
	Args:
		url: A URL de um documento no formato UTF-8.
	"""
	palavras = recuperar_palavras(url)
	imprimir_items(palavras)

if __name__ == '__main__':
	main(sys.argv[1])

# 'http://dinomagri.com/data/aquarela.txt'