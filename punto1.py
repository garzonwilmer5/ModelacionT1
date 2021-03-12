# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 07:39:04 2021

@author: achav
"""

import numpy as np
import matplotlib.pyplot as plt

def dh (h, q, k, A, hv):
    """
    dh (h, q, k, A, hv), dh/dt segun el modelo planteado para los tanques

    Parameters
    ----------
    h : numpy float array 3x1
        altura del fluido en cada tanque.
    q : numpy float array 1x2
        caudal de las bombas A y B.
    k : numpy float array 1x6
        Constante asociada a valvulas.
    A : float
        Area de los tanques.
    hv : float
        Altura de las valvulas.

    Returns
    -------
    dh : numpy float array 3x1
        derivada temporal de las alturas.
    qv : numpy float array 1x6
        cauldal en cada valvula.
    """
    dh = np.empty((3,1))
    qv = np.empty((1,6))
    
    qv[0] = k[0]*np.sqrt(abs(h[0]-h[1]))*np.sign(h[0]-h[1]) #q1-2
    qv[1] = k[1]*np.sqrt(abs(h[2]-h[1]))*np.sign(h[2]-h[1]) #q1-3
    qv[2] = k[2]*np.sqrt(max(0,h[2]-hv)) #q3
    qv[3] = k[3]*np.sqrt(h[0]) #qe1
    qv[4] = k[4]*np.sqrt(h[1]) #qe2
    qv[5] = k[5]*np.sqrt(h[2]) #qe3
    
    dh[0] = q[0] - qv[0] - qv[3]
    dh[1] = qv[0] + qv[1] - qv[4]
    dh[2] = q[1] - qv[1] - qv[2] - qv[5]
    
    dh = dh/A
    return (dh,qv)



def RK(f,dx,x,param):
    """
    RK(f,dx,x,param), Método de Runge Kutta de cuarto orden
    
    Parameters
    ----------
    f : function
        Funcion que describe la derivada de la variable x.
    dx : float
        Paso del metodo.
    x : numpy float array nx1
        valor de la iteracion actual.
    param : diccionario
        Contiene otros parametros de la funcion f.

    Returns
    -------
    x_i : numpy float array nx1
        Valor de x actualizado.
    otro : ?
        Otros elementos retornados por f.
    """
    x = x.reshape((-1,1))
    
    k1,otro = f(x,**param)
    k2,otro = f(x+0.5*k1*dx,**param)
    k3,otro = f(x+0.5*k2*dx,**param)
    k4,otro = f(x+k3*dx,**param)

    x_i = x +0.16667 * dx *(k1 + 2*k2 + 2*k3 +k4)
    
    return x_i,otro


#"Parámetros del sistema"
g = 9.8
St = 1
S = np.ones(6)*0.2

#"Constante de áreas"
k = np.sqrt(2*g)*S/St

#"Alturas"
h = np.zeros((3,1))
hv = 0.1

#Area tanques
A = 1

#Entradas

q = np.array([1, 1])
#paso RK
dx = 1e-2 

ht = np.empty((3,100))
ht[:,0] = h[:,0]
t=range(100)

for i in range(1,100):
    param = {"q":q,"k":k, "A":A, "hv":hv}
    haux,qv =RK(dh,dx,ht[:,i-1],param)
    ht[:,i] = haux.reshape((3))
    
plt.plot(t, ht[0,:],t, ht[1,:],t, ht[2,:])
