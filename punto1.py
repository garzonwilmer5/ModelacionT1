# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 07:39:04 2021

@author: achav
"""

import numpy as np
import matplotlib.pyplot as plt

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
        
    

def dh (h, q, k, A, hv,h_min,h_max):
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
    qv : numpy float array 6x1
        cauldal en cada valvula.
    """
    
    ajustar_limites(h_min,h_max,h.reshape((3)))
    
    dh = np.empty((3,1))
    qv = np.empty((6,1))
    
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
    x = x.reshape((-1,1))
    
    k1,otro = f(x,**param)
    k2,otro = f(x+0.5*k1*dx,**param)
    k3,otro = f(x+0.5*k2*dx,**param)
    k4,otro = f(x+k3*dx,**param)

    x_i = x +0.16667 * dx *(k1 + 2*k2 + 2*k3 +k4)
    
    return x_i,otro


#"Parámetros del sistema"
g = 9.8#m/s^2
#             k1 k2 k3 k4   k5 k6
S = np.array([0.0005, 0.0005, 0, 0.0, 0.0, 0.0])

#"Constante de áreas"
k = np.sqrt(2*g)*S

#Aturas iniciales
h = np.array([[0.5,0.2,0.3]]).T#m
#altura de las valvulas
hv = 0.0#m
#Limites de las alturas
h_min = np.zeros(3)#m
h_max = np.ones(3)*0.5#m

#Area tanques
A = np.pi*0.05**2 #m^2

#Entradas bombas
q = np.array([0.0, 0.0])#m^3/s

#paso RK
dt = 1e-2 #s
#numero de pasos
n = 500

# historico niveles
ht = np.empty((3,n))
ht[:,0] = h[:,0]
#historico caudales
qvt = np.empty((6,n))
qvt[:,0] = np.zeros_like(qvt[:,0])
t=np.arange(n)*dt#s

for i in range(1,n):
    param = {"q":q,"k":k, "A":A, "hv":hv,"h_min":h_min,"h_max":h_max}
    haux,qv =RK(dh,dt,ht[:,i-1],param)
    ht[:,i] = ajustar_limites(h_min,h_max,haux.reshape((3)))
    qvt[:,i] = qv.reshape((6))
    
ht *= 100 #m->cm
qvt *= 1000
#plt.subplot(1, 2, 1)
plt.plot(t, ht[0,:],t, ht[1,:],t, ht[2,:])
plt.legend(["$h_1$","$h_2$","$h_3$"])
plt.xlabel("t (s)")
plt.ylabel("Nivel (cm)")
plt.show()
#plt.subplot(1, 2, 2)
plt.plot(t, qvt[0,:],t, qvt[1,:],t, qvt[2,:],t, qvt[3,:],t, qvt[4,:],t, qvt[5,:])
plt.legend(["$q_{1-2}$","$q_{3-2}$","$q_3$","$q_{e1}$","$q_{e2}$","$q_{e3}$"])
plt.xlabel("t (s)")
plt.ylabel("Caudal $(10^{-3}\ m^3/s)$")

plt.show()
