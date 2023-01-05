# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 17:19:57 2023

@author: Pervin
"""


#Hatalar / istisnalar (exceptions)

#ZeroDivisionError hatasi
a = 10
b = 0

a/b


try:
    print(a/b)
except ZeroDivisionError:
    print("Payda da sifir olmaz")


#tip hatasi
    
a = 10    
b = "2"

a / b

try:
    print(a/b)
except TypeError:
    print("Sayi ve string problemi")




a = 10    
b = 2

a / b

try:
    print(a/b)
except TypeError:
    print("Sayi ve string problemi")



a = [13, 10, 15, 12, 17, 12, 19, 18, 11, 12, 190]
a.sort()
a


a
import numpy as np
np.median(a)