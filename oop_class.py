# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 18:45:01 2023

@author: Pervin
"""

 

#Sinif Ozellikleri (Class attributes)

class VeriBilimci():
    bolum = ''
    sql = 'Evet'
    deneyim_yili = 0
    bildigi_diller = []
    
#Siniflarin ozelliklerine erismek
VeriBilimci.bolum
VeriBilimci.sql

#siniflarin ozelliklerini degistirmek
VeriBilimci.sql = "Hayir"
VeriBilimci.sql

#Sinif Orneklendirmesi (instantiation)

ali = VeriBilimci()

ali.sql
ali.deneyim_yili
ali.bolum
ali.bildigi_diller.append("Python")
ali.bildigi_diller

veli = VeriBilimci()
veli.sql

veli.bildigi_diller

#Ornek ozellikleri

class VeriBilimci():
    bildigi_diller = ["R","PYTHON"]
    bolum = ''
    sql = ''
    deneyim_yili = 0
    def __init__(self):
        self.bildigi_diller = []
        self.bolum = ''

ali = VeriBilimci()
ali.bildigi_diller

veli = VeriBilimci()
veli.bildigi_diller

ali.bildigi_diller.append("Python")
ali.bildigi_diller

veli.bildigi_diller
veli.bildigi_diller.append("R")
veli.bildigi_diller

VeriBilimci.bildigi_diller
ali.bolum

VeriBilimci.bolum
ali.bolum = "istatistik"
VeriBilimci.bolum
veli.bolum
veli.bolum = "end_muh"
veli.bolum
ali.bolum
VeriBilimci.bolum
