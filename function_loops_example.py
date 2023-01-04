# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 18:18:30 2023

@author: Pervin
"""

  
#mini uygulama
#if, for ve fonksiyonlari birlikte kullanmak    

    
maaslar = [1000,2000,3000,4000,5000]

def maas_ust(x):
    print(x*10/100 + x)

def maas_alt(x):
    print(x*20/100 + x)

for i in maaslar:
    if i >= 3000:
        maas_ust(i)
    else:
        maas_alt(i)