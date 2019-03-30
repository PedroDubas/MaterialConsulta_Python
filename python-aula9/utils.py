#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **Descrição**: Funções que auxiliam na fase de coleta, processamento, criação dos classificadores, além de serializar e desserializar objetos.

# Importar as bibliotecas necessárias
import pickle
import nltk
import string
import tweepy
import json
from time import time
import numpy as np
import pandas as pd
import random
import re

# Realizar o download das funcionalidades necessárias do NLTK
nltk.download('stopwords')
nltk.download('rslp')

# Necessário para função de apply_feature do NLTK.
global palavras_sem_repeticao

def tratar_texto(texto):
    string_sem_url = re.sub(r"http\S+", "", str(texto))
    string_sem_user = re.sub(r"@\S+", "", str(string_sem_url))
    string_sem_rt = re.sub(r"RT+", "", str(string_sem_user))
    return str(string_sem_rt).strip()

def remover_pontuacao(base):
    """Essa função remove as pontuações da base.
    Args:
        base: contém todos tweets no formato (texto,classe).
    Returns:
        base_dados: É uma lista de tuplas.
    """
    frases_final = []
    for (frase, classe) in base:
        sem_pontuacao = []
        # Para cada palavra na frase
        frase = tratar_texto(frase)
        for p in frase:
            # Verifica se não é uma pontuação
            if p not in string.punctuation:
                sem_pontuacao.append(p)
        # Refaz a frase
        aux = ''.join(sem_pontuacao)
        # Salva na lista final no formato (texto,classe)
        frases_final.append((aux.lower(), classe))
    # Retorna todo o conjunto sem as pontuações
    return (frases_final)

def remover_stopwords(frases_sem_pontuacao):
    """Essa função remove as stopwords da base.
    Args:
        frases_sem_pontuacao: É uma lista de tuplas.
    Returns:
        frases_final: É uma lista de tuplas (texto, classe).
    """
    stopwordsnltk = nltk.corpus.stopwords.words('portuguese')
    frases_final = []
    for (frase, classe) in frases_sem_pontuacao:
        sem_stop = []
        for palavra in frase.split():
            if palavra not in stopwordsnltk:
                sem_stop.append(palavra)  
        frases_final.append((sem_stop, classe))
    return frases_final

def aplicar_stemmer(frases_sem_stopwords):
    """Função que reduz a palavra ao seu radical
    Args:
        frases_sem_stopwords: lista de tuplas.
    Returns:
        frases_stemming: lista de tuplas.
    """
    stemmer = nltk.stem.RSLPStemmer()
    frases_stemming =[]
    for (frase, classe) in frases_sem_stopwords:
        com_stemming = []
        # Para cada palavra na frase, aplicar o stemmer e salvar
        for palavra in frase:
            com_stemming.append(str(stemmer.stem(palavra)))
        frases_stemming.append((com_stemming, classe))    
    # Retornar todo o conjunto com o stemming aplicado
    return frases_stemming


def extrair_palavras(frases_com_stemmer):
    """Função que unifica todas as palavras do conjunto de dados em uma única lista.
    Args:
        frases_com_stemmer: Frases com o Stemmer já aplicados.
    Returns:
        todas_palavras: lista com todas as palavras.
    """
    todas_palavras = []
    for (palavras, classe) in frases_com_stemmer:
        todas_palavras.extend(palavras)
    return todas_palavras


def aplicar_frequencia(palavras_sem_classe):
    """Função que aplica a frequencia das palavras
    Args:
        palavras_sem_classe: palvras sem a classificação.
    Returns:
        palavras: FreqDist
    """
    palavras = nltk.FreqDist(palavras_sem_classe)
    return palavras


def extrair_palavras_unicas(frequencia_palavras):
    """Função que retorna as palavras únicas
    Args:
        frequencia_palavras: dicionário com a frequencia das palavras.
    Returns:
        freq: palavras unicas.
    """
    freq = frequencia_palavras.keys()
    return freq


def criar_caracteristicas(documento):
    """Função que cria as características do documento, verificando se a palavra existe ou não no documento.
    Args:
        documento: lista com todas as palavras
    Returns:
        caracteristicas: dicionário com as características.
    """
    global palavras_sem_repeticao
    doc = set(documento)
    caracteristicas = {}
    # Para cada palavra
    for palavra in palavras_sem_repeticao:
        # Se a palavra existir no documento é atribuido True, caso contrário False.
        caracteristicas[palavra] = (palavra in doc)    
    # Listar com as caracteristicas da palavra
    return caracteristicas


def estruturar_dados(base):
    """Dada uma base de dados, é realizada toda a estruturação das bases.
    Args:
        base: contém todos tweets no formato (texto,classe).
    Returns:
        base_final: conjunto de dados estruturados.
    """
    global palavras_sem_repeticao
    # Aplicar as funções previamente definidas
    frases_sem_pontuacao = remover_pontuacao(base)
    frases_sem_stopwords = remover_stopwords(frases_sem_pontuacao)
    frases_com_stemmer = aplicar_stemmer(frases_sem_stopwords)
    palavras_sem_classe = extrair_palavras(frases_com_stemmer)
    frequencia_palavras = aplicar_frequencia(palavras_sem_classe)
    palavras_sem_repeticao = extrair_palavras_unicas(frequencia_palavras)
    base_final = nltk.classify.apply_features(criar_caracteristicas, frases_com_stemmer)
    # Retornar os dados estruturados para serem utilizados pela função NLTK.
    return (base_final)

def criar_modelo(treino):
    """Retorna o modelo criado com base no conjunto de treino.
    Args: 
        treino: Base de treino.
    Return:
        modelo: Modelo criado.
    """
    return nltk.NaiveBayesClassifier.train(treino)


def classificar_texto(modelo, frase):
    """Retornar um dicionário com a classe classificada e as probabilidade de todas as classe.
    Args:
        modelo: utilizado para realizar a classificação
        frase: utilizada na classificação
    Returns:
        aux: classe e probabilidades
    """
    aux = {'classe' : None}
    # Realizar a classificação da frase
    aux['classe'] = modelo.classify(frase)
    # Recuperar a probabilidade da frase
    distribuicao = modelo.prob_classify(frase)
    # Para cada classe recuperar a probabilidade
    for classe in distribuicao.samples():
        aux[classe] = distribuicao.prob(classe)
    # Retornar a estrutura com a classe e as probabilidades
    return aux
