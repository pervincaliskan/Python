# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:33:52 2023

@author: Pervin
"""


#Setler - Klasik Kume Islemleri

# =============================================================================
# difference() ile iki kumenin farkini ya da "-" ifadesi
# intersection() iki kume kesisimi ya da "&" ifadesi
# union() iki kumenin birlesimi
# symmetric_difference() ikisinde de olmayanlari.
# =============================================================================


#difference

set1 = set([1,3,5])
set2 = set([1,2,3])

set1.difference(set2)

set2.difference(set1)

set1.symmetric_difference(set2)

set1 - set2
set2 - set1

#intersection & union

set1 = set([1,3,5])
set2 = set([1,2,3])

set1.intersection(set2)
set2.intersection(set1)


kesisim = set1 & set2
kesisim

birlesim = set1.union(set2)
birlesim

set1.intersection_update(set2)
set1


#Set Sorgu Islemleri

set1 = set([7,8,9])
set2 = set([5,6,7,8,9,10])

#iki kumenin kesisiminin bos olup 
#olmadiginin sorgulanması

set1.isdisjoint(set2)


#bir kumenin butun elemanlarinin baska bir kume
#icerisinde yer alip almadigi

set1.issubset(set2)

#bir kumenin bir diger kumeyi kapsayip kapsamadigi

set2.issuperset(set1)




dir(str)
