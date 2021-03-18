#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 07:04:27 2021

@author: wilmer
"""

import numpy as np
import matplotlib.pyplot as plt

#contantes
R_j = 8.314472 #J/(mol.K) 
R_cal = 1.987 #cal/(mol.K) 

#tabla reaccion hacia adelante
#[A(cm^3/mol.s),b,E(cal/mol)]
table_f = np.array([[1.915e14,0,1.644e4],
                    [5.08e4,2.67,6.292e3],
                    [2.16e8,1.51,3.43e3],
                    [2.97e6,2.02,1.34e4],
                    [1.66e13,0,8.23e2],
                    [3.25e13,0,0]])
#tabla reaccion hacia atras
#[A(cm^3/mol.s),b,E(cal/mol)]
table_b = np.array([[5.481e11,0.39,-2.93e2],
                    [2.667e4,2.65,4.88e3],
                    [2.298e9,1.4,1.832e4],
                    [1.465e5,2.11,-2.904e3],
                    [3.164e12,0.35,5.551e4],
                    [3.252e12,0.33,5.328e4]])

def arrhenius(T,A,b,E):
    global R_cal
    aux = np.exp(-E/(R_cal*T))
    k = A*(T**b)*aux
    return k

def dx(x):
    global table_f, table_b
    k_f = arrhenius(x[7],table_f[:,0],table_f[:,1],table_f[:,2])
    k_b = arrhenius(x[7],table_b[:,0],table_b[:,1],table_b[:,2])
    dx= np.empty((8))
    #[0   1   2  3   4    5    6  7]
    #[O2, H2, H, OH, H2O, HO2, O, T]
    
    react = np.array([[k_f[0]*x[2]*x[0] , k_b[0]*x[6]*x[3]],
                      [k_f[1]*x[6]*x[1] , k_b[1]*x[2]*x[3]],
                      [k_f[2]*x[3]*x[1] , k_b[2]*x[2]*x[4]],
                      [k_f[3]*x[6]*x[4] , k_b[3]*x[3]*x[1]],
                      [k_f[4]*x[5]*x[2] , k_b[4]*x[1]*x[0]],
                      [k_f[5]*x[5]*x[6] , k_b[5]*x[3]*x[0]]])
    #dO2/dt
    dx[0] = (-react[0,0] + react[0,1]
             +react[4,0] - react[4,1]
             +react[5,0] - react[5,1])
    #dH2/dt
    dx[1] = (-react[1,0] + react[1,1]
             -react[2,0] + react[2,1]
             +react[4,0] - react[4,1])
    #dH/dt
    dx[2] = (-react[0,0] + react[1,1]
             +react[1,0] - react[1,1]
             +react[2,0] - react[2,1]
             -react[4,0] + react[4,1])
    #dOH/dt
    dx[3] = (react[0,0] - react[0,1]
            +react[1,0] - react[1,1]
            -react[2,0] + react[2,1]
            +react[3,0] - react[3,1]) 
    #dH2O/dt
    dx[4] = (react[2,0] - react[2,1]
            -react[3,0] + react[3,1]) 
    #dHO2/dt
    dx[5] = (-react[4,0] + react[4,1]
            -react[5,0] + react[5,1])
    #dO/dt
    dx[6] = (react[0,0] - react[0,1]
            -react[1,0] + react[1,1]
            -react[3,0] + react[3,1]
            -react[5,0] + react[5,1])
    #dT/dt
    dx[7] = -1/(x[2]+1e-6) * dx[1]*x[7]
    
    return dx

def RK(f,dx,x,param={}):
    """
    RK(f,dx,x,param), Método de Runge Kutta de cuarto orden
    
    Parameters
    ----------
    f : function
        Funcion que describe la derivada de la variable x.
    dx : float
        Paso del metodo.
    x : numpy float array nx1
        valor de la iteracion actual.d
    param : diccionario
        Contiene otros parametros de la funcion f.

    Returns
    -------
    x_i : numpy float array nx1
        Valor de x actualizado.
    otro : ?
        Otros elementos retornados por f.
    """    
    k1 = f(x,**param)
    k2 = f(x+0.5*k1*dx,**param)
    k3 = f(x+0.5*k2*dx,**param)
    k4 = f(x+k3*dx,**param)

    x_i = x +0.16667 * dx *(k1 + 2*k2 + 2*k3 +k4)
    
    return x_i
def ajustar_limites(bd,bu,arr):
    """
    ajustar_limites(bd,bu,arr), garantiza que todos los elementos del vector
    dentro de los limites establecidos

    Parameters
    ----------
    bd : numpy float array nxm
        limite inferior.
    bu : numpy float array nxm
        limite superior.
    arr : numpy float array nxm
        arreglo a ajustar.

    Returns
    -------
    arr: float array
        arreglo con los limites ajustados
    """
        
    ibd=arr<bd
    ibu=arr>bu
    arr[ibd] = bd[ibd]
    arr[ibu] = bu[ibu]
    
    return arr

#A(cm^3/mol.s)->A(m^3/mol.s)
table_f[:,0]*=1e-6
table_b[:,0]*=1e-6

#Presion
P = 1.2*101325#Pa
#condicon inicial [O2, H2, H, OH, H2O, HO2, O, T]
x0 = np.zeros((8))
x0[7] = 1200 #K
x0[0] = 0.3/1.3*P/(x0[7]*R_j) # concetracion O2
x0[1] = 1/1.3*P/(x0[7]*R_j) # concetracion H2

#Limites varaibles
h_min = np.zeros(8)
h_max = np.ones(8)*np.inf

#paso RK
dt = 1e-12 #s
#numero de pasos
n = int(1*1e6)

#historico variabales
xh = np.empty((8,n))
xh[:,0] = x0
t=np.arange(n)#nano s

for i in range(1,n):
    xh[:,i]=ajustar_limites(h_min,h_max,RK(dx,dt,xh[:,i-1]))
t = t*1e-6     
plt.plot(t, xh[0,:],t, xh[1,:],t, xh[2,:],t, xh[3,:],t, xh[4,:],t, xh[5,:],t, xh[6,:])
plt.legend(["$O_2$","$H_2$","$H$","$OH$","$H_2O$","$HO_2$","$O$"])
plt.xlabel("t ($\mu$s)")
plt.ylabel("Concentracion")
plt.show()
plt.plot(t, xh[7,:])
plt.xlabel("t ($\mu$s)")
plt.ylabel("T(C)")
plt.show()






