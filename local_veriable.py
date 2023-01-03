# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 18:10:04 2023

@author: Pervin
"""


#Local ve Global Degiskenler

x = 10
y = 25

def carpma_yap(x = 2,y = 1):
    return x*y


carpma_yap(2,3)

#Local Etki Alanindan Global Etki Alanini Degistirmek


x = []
#del x

def eleman_ekle(y):
    x.append(y)
    print(str(y) + " ifadesi eklendi")

eleman_ekle("ali")

eleman_ekle("veli")

x