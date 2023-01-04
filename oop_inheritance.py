# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 20:42:58 2023

@author: Pervin
"""


#Miras Yapilari (inheritance)

class Employees():
    def __init__(self):
        self.FirstName = ""
        self.LastName = ""
        self.Address = ""

class DataScience(Employees):
    def __init__(self):
        self.Programming = ""

veribilimci1 = DataScience()
#veribilimci1.

class Marketing(Employees):
    def __init__(self):
        self.StoryTelling = ""

mar1 = Marketing()
#mar1.


class Employee_yeni():
    def __init__(self, FirstName,LastName,Address):
        self.FirstName = FirstName
        self.LastName = LastName
        self.Address = Address


ali = Employee_yeni("a","b","c")
ali.FirstName

