# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 07:39:04 2021

@author: achav
"""

import numpy as np


"Parámetros del sistema"
g = 9.8
St = 1
S = np.ones(6)

"Constante de áreas"
k = np.sqrt(2*g)*S/St

"Alturas"
h = np.zeros(3)

"Entradas"

q = np.array([1, 1])

"Definición de la ecuación diferencial"

def dh (q, k, h):
    dh1 = q[0] - k[0]*np.sqrt(abs(h[0]-h[1]))*np.sign(h[0]-h[1])-k[3]*np.sqrt(h[0])
    dh2 = k[0]*np.sqrt(abs(h[0]-h[1]))*np.sign(h[0]-h[1]) + k[1]*np.sqrt(h[2]-h[1])-np.sign(h[2]-h[1]) - k[4]*np.sqrt(h[1])
    dh3 = q[1] - k[1]*np.sqrt(abs(h[2]-h[1]))*np.sign(h[2]-h[1])-k[2]*np.sqrt(h[2])-k[2]*np.sqrt(h[2])
    dh = np.array([[dh1, dh2, dh3]]).T
    return dh

dx = 1e-2

"Método de Runge Kutta"

