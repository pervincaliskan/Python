# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:32:21 2023

@author: Pervin
"""

# Veri Yapıları - Dictionary (Sözlük)


#Sozluk Olusturma
sozluk = {"REG" : "Regresyon Modeli",
          "LOJ" : "Lojistik Regresyon",
          "CART" : "Classification and Reg"}

sozluk

len(sozluk)

sozluk = {"REG" : 10,
          "LOJ" : 20,
          "CART" : 30}

sozluk


sozluk = {"REG" : ["RMSE",10],
          "LOJ" : ["MSE", 20],
          "CART" : ["SSE",30]}

sozluk

#Sozluk Eleman Islemleri

sozluk = {"REG" : "Regresyon Modeli",
          "LOJ" : "Lojistik Regresyon",
          "CART" : "Classification and Reg"}


sozluk[0]

sozluk["REG"]
sozluk["LOJ"]

sozluk = {"REG" : ["RMSE",10],
          "LOJ" : ["MSE", 20],
          "CART" : ["SSE",30]}


sozluk["REG"]

sozluk = {"REG" : {"RMSE": 10,
                   "MSE" : 20,
                   "SSE" : 30},

          "LOJ" : {"RMSE": 10,
                   "MSE" : 20,
                   "SSE" : 30},
                   
          "CART" : {"RMSE": 10,
                   "MSE" : 20,
                   "SSE" : 30}}

sozluk
sozluk["REG"]["SSE"]

#Sozluk - Eleman Eklemek & Degistirmek

sozluk = {"REG" : "Regresyon Modeli",
          "LOJ" : "Lojistik Regresyon",
          "CART" : "Classification and Reg"}

sozluk["GBM"] = "Gradient Boosting Mac"
sozluk

sozluk["REG"] = "Coklu Dogrusal Regresyon"
sozluk

sozluk[1] = "Yapay Sinir Aglari"

sozluk

l = [1]
l

sozluk[l] = "yeni bir sey"

t = ("tuple",)

sozluk[t] = "yeni bir sey"
sozluk