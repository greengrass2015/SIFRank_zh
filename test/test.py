#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge_sy"
# Date: 2020/2/21

from embeddings import sent_emb_sif, word_emb_elmo
from model.method import SIFRank, SIFRank_plus
import thulac
import jieba.analyse

#download from https://github.com/HIT-SCIR/ELMoForManyLangs
model_file = r'../auxiliary_data/zhs.model/'

ELMO = word_emb_elmo.WordEmbeddings(model_file)
SIF = sent_emb_sif.SentEmbeddings(ELMO, lamda=1.0)
#download from http://thulac.thunlp.org/
zh_model = thulac.thulac(model_path=r'../auxiliary_data/thulac.models/',user_dict=r'../auxiliary_data/user_dict.txt')
elmo_layers_weight = [0.0, 1.0, 0.0]

text = "配有第六代处理器，商务办公或普通娱乐游戏都能应对，2G独立显卡将性能显著提升，图像处理清晰，软件运行不卡顿。15.6英寸的背光高清显示屏能为你提供绚丽的色彩画面。机身纤薄出众。"
keyphrases = SIFRank(text, SIF, zh_model, N=15, elmo_layers_weight=elmo_layers_weight)
keyphrases_ = SIFRank_plus(text, SIF, zh_model, N=15, elmo_layers_weight=elmo_layers_weight)
print("------------------------------------------")
print("原文:"+text)
print("------------------------------------------")
print("SIFRank_zh结果:")
print(keyphrases)
print("SIFRank+_zh结果:")
print(keyphrases_)
print("------------------------------------------")
print("jieba分词TFIDF算法结果:")
print(jieba.analyse.tfidf(text, topK=15, withWeight=True, allowPOS=('n','nr','ns')))
print("jieba分词TFIDF算法结果:")
print(jieba.analyse.textrank(text, topK=15, withWeight=True, allowPOS=('n','nr','ns')))