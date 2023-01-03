# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 18:10:59 2023

@author: Pervin
"""


# KARAR & KONTROL YAPILARI


#True-False Sorgulamaları

sinir = 5000

sinir == 4000

sinir == 5000

sinir == 5000


5 == 4

5 == 5


#if

sinir = 50000
gelir = 60000

gelir < sinir

if gelir < sinir:
    print("Gelir sinirdan kucuk")
    
if gelir > sinir:
    print("Gelir sinirdan kucuk")
  

#else

sinir = 50000
gelir = 35000
    
if gelir > sinir:
    print("Gelir sinirdan buyuk")
else:
    print("Gelir sinirdan kucuk")    

#diger ornek
sinir = 50000
gelir = 50000
    
if gelir == sinir:
    print("Gelir sinira esittir")
else:
    print("Gelir sinira esit degildir")  

#elif

sinir = 50000
gelir1 = 60000
gelir2 = 50000
gelir3 = 35000
    
if gelir1 > sinir:
    print("Tebrikler, hediye kazandiniz.")
elif gelir1 < sinir:
    print("Uyari!")
else:
    print("Takibe devam")  


if gelir3 > sinir:
    print("Tebrikler, hediye kazandiniz.")
elif gelir3 < sinir:
    print("Uyari!")
else:
    print("Takibe devam")  


if gelir2 > sinir:
    print("Tebrikler, hediye kazandiniz.")
elif gelir2 < sinir:
    print("Uyari!")
else:
    print("Takibe devam")  