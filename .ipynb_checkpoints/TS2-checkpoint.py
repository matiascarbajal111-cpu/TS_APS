# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 11:59:57 2025

@author: Matias
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ----------------------
# Parámetros
# ----------------------
N = 200        # número de muestras
f = 2000        # frecuencia de la señal
fs = 50000      # frecuencia de muestreo
fsquare = 4000  # frecuencia cuadrada

t = np.arange(N) / fs

# ----------------------
# Señales
# ----------------------
s1 = np.sin(2 * np.pi * f * t)
s2 = 2 * np.sin(2 * np.pi * f * t + np.pi/2)
s3 = np.sin(2 * np.pi * (f/2) * t)
sq = signal.square(2 * np.pi * fsquare * t)

# ----------------------
# Función ecuación en diferencias
# ----------------------
def ecuacion_diferencias(x, b, a):
    """
    Implementa la ecuación en diferencias:
        y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + a[0]*y[n-1] + a[1]*y[n-2] + ...
    
    x : array_like -> señal de entrada
    b : list -> coeficientes de entrada (x[n], x[n-1], ...)
    a : list -> coeficientes de realimentación (y[n-1], y[n-2], ...)
    """
    N = len(x)
    y = np.zeros(N)
    M = len(b)
    K = len(a)
    
    for n in range(N):
        # Parte de la entrada
        for i in range(M):
            if n-i >= 0:
                y[n] += b[i] * x[n-i]
        # Parte de la realimentación
        for j in range(1, K+1):
            if n-j >= 0:
                y[n] += a[j-1] * y[n-j]
    return y

def energia_potencia(y):
    E = np.sum(y**2)
    P = np.mean(y**2)
    return E, P


# ----------------------
# Coeficientes
# ----------------------
b = [0.03, 0.05, 0.03]   # coeficientes de entrada
a = [1.5, -0.5]          # coeficientes de realimentación

# ----------------------
# Respuesta al impulso
# ----------------------

delta = np.zeros(N)
delta[0] = 1.0
h = ecuacion_diferencias(delta, b, a)
# ----------------------
# Salida con entrada s1
# ----------------------
y_s1 = ecuacion_diferencias(s1, b, a)

# ----------------------
# Salida con entrada s2
# ----------------------
y_s2 = ecuacion_diferencias(s2,b,a)
# ----------------------
# Salida con entrada s3
# ----------------------
y_s3 = ecuacion_diferencias(s3,b,a)
# ----------------------
# Salida con entrada sq
# ----------------------
y_sq = ecuacion_diferencias(sq,b,a)

# Salida con entrada s1 con convolucion de h[n] e y[n]
# ----------------------





plt.plot(t, y_s1, marker='o', linestyle='-')
plt.title("Salida con entrada s1[n]")
plt.xlabel("t[s]")
plt.ylabel("y[n]")
plt.grid(True)
plt.show()

plt.plot(t, y_s2, marker='o', linestyle='-')
plt.title("Salida con entrada s2[n]")
plt.xlabel("t[s]")
plt.ylabel("y[n]")
plt.grid(True)
plt.show()

plt.plot(t, y_s3, marker='o', linestyle='-')
plt.title("Salida con entrada s3[n]")
plt.xlabel("t")
plt.ylabel("y[n]")
plt.grid(True)
plt.show()

plt.plot(t, y_sq, marker='o', linestyle='-')
plt.title("Salida con entrada sq[n]")
plt.xlabel("t[s]")
plt.ylabel("y[n]")
plt.grid(True)
plt.show()

plt.plot(t, h, marker='o', linestyle='-')
plt.title("Respuesta al impulso h[n]")
plt.xlabel("t[s]")
plt.ylabel("h[n]")
plt.grid(True)
plt.show()


ysc1_np = np.convolve(s1, h)[:N]  # cortamos a la longitud original

# Graficar comparación
plt.plot(t, y_s1, marker='o', linestyle='-', label='Ecuación diferencias')
plt.plot(t, ysc1_np, marker='x', linestyle='--', label='Convolución np.convolve')
plt.title("Comparación salida s1[n]")
plt.xlabel("t [s]")
plt.ylabel("y[n]")
plt.grid(True)
plt.legend()
plt.show()

#2---------------
b = [ 1,0,0,0,0,0,0,0,0,0,3]
a = [0 , 0]
    
y__s1 = ecuacion_diferencias(delta, b, a)


plt.plot(t, y__s1, marker='o', linestyle='-')
plt.title("Salida con entradaa s1[n]")
plt.xlabel("t[s]")
plt.ylabel("y[n]")
plt.grid(True)
plt.show()

b = [ 1]
a = [ 0,0,0,0,0,0,0,0,0,0,3]
    
y___s1 = ecuacion_diferencias(delta, b, a)


plt.plot(t, y___s1, marker='o', linestyle='-')
plt.title("Salida con entradaa s1[n]")
plt.xlabel("t[s]")
plt.ylabel("y[n]")
plt.grid(True)
plt.show()