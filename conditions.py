# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 18:16:35 2023

@author: Pervin
"""


#mini uygulama
sinir = 50000
magaza_adi = input("Magaza Adi Nedir?")
gelir = int(input("Gelirinizi Giriniz:"))

if gelir > sinir:
    print("Tebrikler:" + magaza_adi + " promosyon kazandiniz!")
elif gelir < sinir:
    print("Uyari! Cok dusuk gelir:" + str(gelir))
else:
    print("Takibe devam")