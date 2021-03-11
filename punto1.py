# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 07:39:04 2021

@author: achav
"""

import numpy as np
import matplotlib.pyplot as plt

"Definición de la ecuación diferencial"

def dh (q, k, h):
    dh1 = q[0] - k[0]*np.sqrt(abs(h[0]-h[1]))*np.sign(h[0]-h[1])-k[3]*np.sqrt(h[0])
    dh2 = k[0]*np.sqrt(abs(h[0]-h[1]))*np.sign(h[0]-h[1]) + k[1]*np.sqrt(abs(h[2]-h[1]))-np.sign(h[2]-h[1]) - k[4]*np.sqrt(h[1])
    dh3 = q[1] - k[1]*np.sqrt(abs(h[2]-h[1]))*np.sign(h[2]-h[1])-k[2]*np.sqrt(h[2])-k[2]*np.sqrt(h[2])
    dh = np.array([dh1, dh2, dh3])
    return dh



"Método de Runge Kutta"

def RK(f,h,y,q,k):
    y = y.reshape((-1,1))
    
    k1 = f(q,k,y)
    k2 = f(q,k,y+0.5*k1*h)
    k3 = f(q,k,y+0.5*k2*h)
    k4 = f(q,k,y+k3*h)

    y_i = y +0.16667 * h *(k1 + 2*k2 + 2*k3 +k4)
    
    return y_i


#"Parámetros del sistema"
g = 9.8
St = 1
S = np.ones(6)

#"Constante de áreas"
k = np.sqrt(2*g)*S/St

#"Alturas"
h = np.zeros((3,1))

#Entradas

q = np.array([1, 1])
#paso RK
dx = 1e-2 

ht = np.empty((3,100))
ht[:,0] = h[:,0]
t=range(100)

for i in range(1,100):
    ht[:,i]=RK(dh,dx,ht[:,i-1],q,k).reshape((3))
    
plt.plot(t, ht[0,:],t, ht[1,:],t, ht[2,:])
