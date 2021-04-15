#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 09:23:54 2021

@author: wilmer
"""

import numpy as np
import matplotlib.pyplot as plt

g = 9.81  # m/s^2
amp = 0  # N
mf = 0.5  # Hz


def w(k6, k5, mt, m6, wn):
    r = np.sqrt(((k5+k6)/m6)**2 - 2*(k5**2 + k5*k6)/(mt*m6)-3*(k5/mt)**2)
    f = -0.5*((k5+k6)/m6-k5/mt - r)
    f += wn**2
    return f


def dw(k6, k5, mt, m6, wn, h=1e-6):
    return (w(k6+h, k5, mt, m6, wn)-w(k6-h, k5, mt, m6, wn))/(2*h)


def newton_raphson(f, df, x_0=0, itmax=100, ex=1e-6, ey=1e-6, param={}):
    """
    Parameters
    ----------
    f : Function
        Funcion a la que se le desea encontrar sus raices.
    df : Funtion
        derivada de f.
    x_0 : float, optional
        Valor semilla. The default is 0.
    itmax : int, optional
        Maximo de iteraciones permitidas. The default is 100.
    ex : float, optional
        tolerancia de error en x. The default is 1e-6.
    ey : float, optional
        tolerancia de error en f. The default is 1e-6.
    param : diccionary, optional
        otros argumentos de f y df. The default is {}.

    Returns
    -------
    x : TYPE
        DESCRIPTION.

    """
    error = 1
    it = 0
    x = x_0
    ef = f(x, **param)
    while it < itmax and (error > ex or abs(ef) > ey):
        dx = ef/df(x, **param)
        x -= dx
        error = abs(dx)
        ef = f(x, **param)
        it += 1
    return x


def model(x, t, m, l, A, I):
    """
    Descripcion modelo generador mareomotriz

    Parameters
    ----------
    x : numpy float array 6x1
        Variables del modelo [V5, Y5, V6, Y6, w, theta].
    t: float
        tiempo
    m : float array 6x1
        Masas de los elementos.
    l : float array 4x1
        longitud brazos tetrapandulo
    A : numpy float array 4x4
        Matriz caracteristica del sistema masa resorte amortiguador
    I : float
        Inercia angular sistema pendular
    Returns
    -------
    dx : numpy float array
         Derivada temporal de las variables.
    """
    global g, amp, mf

    dx = np.empty((6,))

    Fm = amp*np.sin(mf*2*np.pi*t)+amp
    dx[0:4] = A@x[0:4]
    dx[1] -= g
    dx[3] += Fm/m[5] - g
    # dw/dt
    dx[4] = (-0.8/I)*(dx[0]+g)*((l[0]*m[0]-l[2]*m[2])*np.sin(x[5])
                                + (l[1]*m[1]-l[3]*m[3])*np.cos(x[5]))
    # dtheta/dt
    dx[5] = x[4]

    return dx


def RK(f, dt, x, t, param={}):
    """
    RK(f,dt,x,t,param), Método de Runge Kutta de cuarto orden

    Parameters
    ----------
    f : function
        Funcion que describe la derivada de la variable x.
    dt : float
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
    k1 = f(x, t, **param)
    k2 = f(x+0.5*k1*dt, t+0.5*dt, **param)
    k3 = f(x+0.5*k2*dt, t+0.5*dt, **param)
    k4 = f(x+k3*dt, t+dt, **param)

    x_i = x + 0.16667 * dt * (k1 + 2*k2 + 2*k3 + k4)

    return x_i


def simpson_1_3(x, dt):
    """
    Retorna la integral numerica segun regla de Simpson 1/3

    Parameters
    ----------
    x : array/ numpy array
        x = f(t).
    dt : float
        Intervalo de tiempo con que se realizo el muestreo

    Returns
    -------
    I : float
        Resultado de la integral.
    """

    f1_3 = 1.0/3.0

    I = 0
    i = 0
    while i < len(x):
        if i+2 < len(x):
            I += f1_3*dt*(x[i]+4*x[i+1]+x[i+2])
        elif i+1 < len(x):
            I += 0.5*dt*(x[i]+x[i+1])
        i += 2

    return I


def ajustar_limites(bd, bu, arr):
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

    ibd = arr < bd
    ibu = arr > bu
    arr[ibd] = bd[ibd]
    arr[ibu] = bu[ibu]

    return arr


def PSO(f, lb, ub, lr=0.1, w=1, gp=0.1, gg=0.1, population_size=10, maxit=100, minimize=True, params={}):
    """
    Particle Swarm Optimization Algorithm

    Parameters
    ----------
    f : function to optimize
        function (x,**params).
    lb : numpy float array 
        Lower boundaries.
    ub : numpy float array 
        Upper boundaries.
    lr : float, optional
        Learning rate. The default is 0.1.
    w : float, optional
        Inercia factor. The default is 1.
    gp : float, optional
        Atraction factor to best particle position. The default is 0.1.
    gg : float, optional
        Atraction factor to best swarm position. The default is 0.1.
    population_size : int, optional
        Number of particles. The default is 10.
    maxit : int, optional
        Number of iterations. The default is 100.
    minimize : bool, optional
        Indicate if it is a minimization or maximization problem. The default is True.
    params : Diccionary, optional
        Other paramenters of f. The default is {}.

    Returns
    -------
    swarm_best :numpy float array
        Best solution finded by the swarm.

    """
    # Initialize particles
    population = np.random.rand(population_size, len(lb))
    velocity = np.random.rand(population_size, len(lb))
    swarm_best = np.empty((len(lb),))
    pb_value = np.empty((population_size,))

    best_val = np.inf if minimize else -np.inf
    for i in range(population_size):  # fit particles in boundaries
        population[i] = population[i]*(ub-lb) + lb
        velocity[i] = velocity[i]*2*abs(ub-lb) - abs(ub-lb)

        pb_value[i] = f(population[i], **params)
        # update best solution
        if minimize and (pb_value[i] < best_val):
            best_val = pb_value[i]
            swarm_best = population[i]
        if (not minimize) and (pb_value[i] > best_val):
            best_val = pb_value[i]
            swarm_best = population[i]

    particle_best = np.copy(population)
    it = 0
    while it < maxit:
        for i in range(population_size):  # update particles
            # update velocity
            r = np.random.rand(2, len(lb))
            dx_bx = particle_best[i] - population[i]
            dx_bs = swarm_best - population[i]
            velocity[i] = w*velocity[i] + gp*r[0]*dx_bx + gg*r[1]*dx_bs

            # update particle position
            population[i] += lr*velocity[i]
            population[i] = ajustar_limites(
                lb, ub, population[i])  # check boundaries

            # evaluate solution
            val = f(population[i], **params)
            # update best solution
            if minimize:
                if pb_value[i] > val:
                    pb_value[i] = val
                    particle_best[i] = population[i]
                    if best_val > val:
                        best_val = val
                        swarm_best = population[i]

            if not minimize:
                if pb_value[i] < val:
                    pb_value[i] = val
                    particle_best[i] = population[i]
                    if best_val < val:
                        best_val =val
                        swarm_best = population[i]

        print(it, best_val, swarm_best)
        it += 1
    return swarm_best


def aux_foo(x, mf, k, b):
    """
    Funcion intermedia para optimizacion por PSO

    Parameters
    ----------
    x : numpy float array shape (8,)
        Varibles del pendulo, masa y longitudes.
    mf : numpy float array shape (2,)
        Masa mt y m6.
    k : numpy float array shape (2,)
        Contantes de los resortes.
    b : numpy float array shape (2,)
        Contantes de los amortiguadores.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    m = np.empty((6,))
    m[0:4] = x[0:4]
    m[4] = mf[0] - sum(m[0:4])
    m[5] = mf[1]
    l = x[4:8]
    return simulacion(m, l, k, b, plot=False)


def simulacion(m, l, k, b, plot=True):
    # paso RK
    dt = 0.1  # s
    # numero de pasos
    n = int(100/dt)  # simular200 segundos

    # historico variabales
    xh = np.empty((6, n))
    xh[:, 0] = np.zeros((6,))
    t = np.arange(n)*dt

    I = 0
    for i in range(4):
        I += m[i]*(l[i]**2)

    A = np.array([[-b[0], -k[0], b[0], k[0]],
                  [1, 0, 0, 0],
                  [b[0], k[0], -(b[0]+b[1]), -(k[0]+k[1])],
                  [0, 0, 1, 0]])
    mt = sum(m[0:5])
    A[0, :] = A[0, :]/mt
    A[2, :] = A[2, :]/m[5]

    param = {"m": m, "l": l, "A": A, "I": I}

    for i in range(1, n):
        xh[:, i] = RK(model, dt, xh[:, i-1], t[i], param)
        """
        if xh[5,i]>np.pi:
            xh[5,i] -= 2*np.pi
        if xh[5,i]<-np.pi:
            xh[5,i] += 2*np.pi
        """
    score = simpson_1_3(abs(xh[4, :]), dt)
    if plot:
        plt.plot(t, xh[1, :], t, xh[3, :])
        plt.legend(["$y_5$", "$y_6$"])
        plt.xlabel("t (s)")
        plt.ylabel("Posición (m)")
        plt.grid()
        plt.show()
        plt.plot(t, xh[5, :])
        plt.title("score: "+str(score))
        plt.xlabel("t (s)")
        plt.ylabel("Angulo (rad)")
        plt.show()

    

    return score


# parametros del modelo
k = np.array([100, 1000])  # N/m
b = np.array([5, 5])  # kg/s
mt = 10
m6 = 10

k[1] = newton_raphson(w, dw, x_0=k[1], itmax=100, param={
                      "k5": k[0], "mt": mt, "m6": m6, "wn": mf})

params = {"mf": np.array([mt, m6]), "k": k, "b": b}
lb = np.ones((8,))*0.1
ub = np.array([5, 5, 5, 5, 1, 1, 1, 1])
#x = PSO(aux_foo, lb, ub, population_size=20,maxit=200, minimize=False, params=params)
x = [0.1, 0.1, 0.1,4.00116512,0.1,0.1,0.1,0.1]
#x = [0.1, 1.00701334, 3.24425462, 0.1, 0.1, 0.1, 0.1, 0.1]
"""
927.4325918250141 [0.1, 1.71723913, 2.91673739, 0.1, 0.2619616, 0.1, 0.1, 1.]
"""

m = np.empty((6,))
m[0:4] = x[0:4]
m[4] = mt - sum(m[0:4])
m[5] = m6
l = x[4:8]
score = simulacion(m, l, k, b)
