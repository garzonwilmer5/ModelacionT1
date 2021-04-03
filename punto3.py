# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 15:42:37 2021

@author: achav
"""

import numpy as np
import matplotlib.pyplot as plt

def funct(tao1, tao2, r1, r2, k1, k2, b1, b2, x1, x2, i):  
    if(i == 0):
        f = (tao1/r1-k1*x1)/b1
    else: 
        f = (tao2/r2-k2*x2)/b2
    return f

def euler(tao1, tao2, r1, r2, k1, k2, b1, b2, x1, x2, i):
    fi_out = np.zeros(n)
    fi_in = [x1, x2]
    for i in range(n):
        fi_out[i] = fi_in[i] + dt*funct(tao1, tao2, r1, r2, k1, k2, b1, b2, x1, x2, i)
    return fi_out


# Variables del modelo físico

J1 = 8e-4
J2 = 8e-4
tao1 = 3.136
tao2 = 3.136
r1 = 0.03
r2 = 0.03
k1 = 50
k2 = 50
k3 = 200
b1 = 0.2 
b2 = 0.2
omega1 = 100*np.pi
omega2 = 100*np.pi

# Variables del modelo matemático-computacional

n = 2
tf = 0.04
dt = 0.001
it = int(tf/dt)

t = np.zeros(it+1)
x1 = np.zeros(it+1)
x2 = np.zeros(it+1)
x3 = np.zeros(it+1)

x1[0] = 0
x2[0] = 0
x3 = (tao1/r1+tao2/r2)/k3

theta = np.arange(0, np.pi/2, np.pi/100)
x3theta = x3*np.cos(theta)


for i in range(1,it+1):
    t[i] = t[i-1] + dt
    [x1[i], x2[i]] = euler(tao1, tao2, r1, r2, k1, k2, b1, b2, x1[i-1], x2[i-1], i)    


plt.figure(1)

plt.grid()
plt.plot(1000*t, x1, "-r", label="x1")     
plt.plot(1000*t, x2, "-b", label="x2")

plt.legend(loc="upper right")
plt.xlabel("Tiempo [ms]")
plt.ylabel("Deformación de la cinta [m]")

plt.figure(2)

plt.grid()
plt.plot(theta, x3theta, "-g", label="x3")
plt.legend(loc="upper right")
plt.xlabel("Ángulo Theta [rad]")
plt.ylabel("Deformación del resorte [m]")


