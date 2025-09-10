# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 11:59:57 2025

@author: Matias
"""
import numpy as np
import matplotlib.pyplot as plt
# %%
from scipy import signal


# ----------------------
# Parámetros
# ----------------------
# %%
N = 100        # número de muestras

f = 2000       # frecuencia de la señal
fs = 50000     # frecuencia de muestreo
fsquare = 4000 # frecuencia cuadrada

n = np.arange(N)  # eje discreto

# ----------------------
# Señales
# ----------------------
s1 = np.sin(2 * np.pi * f * n / fs)
s2 = 2 * np.sin(2 * np.pi * f * n / fs + np.pi/2)
s3 = np.sin(2 * np.pi * (f/2) * n / fs)
sq = signal.square(2 * np.pi * fsquare * n / fs)

# ----------------------
# Función ecuación en diferencias
# ----------------------
def ecuacion_diferencias(x, b, a):
    N = len(x)
    y = np.zeros(N)
    M = len(b)
    K = len(a)
    for i in range(N):
        # Parte de la entrada
        for j in range(M):
            if i-j >= 0:
                y[i] += b[j]*x[i-j]
        # Parte de la realimentación
        for j in range(1,K+1):
            if i-j >= 0:
                y[i] += a[j-1]*y[i-j]
    return y

def energia_potencia(y):
    E = np.sum(y**2)
    P = np.mean(y**2)
    return E, P

def plot_signal(n, y, title):
    plt.figure()
    plt.plot(n, y, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("desplazamiento [n]")
    plt.ylabel("y[n]")
    plt.grid(True)
    plt.show()

# ----------------------
# Coeficientes
# ----------------------
a = [0.0]*11
b = [0.0]*11
b = [0.03, 0.05, 0.03]
a = [1.5, -0.5]

# ----------------------
# Respuesta al impulso
# ----------------------
delta = np.zeros(N)
delta[0] = 1
h = ecuacion_diferencias(delta, b, a)
plot_signal(n, h, "Respuesta al impulso h[n]")

# ----------------------
# Salidas
# ----------------------
y_s1 = ecuacion_diferencias(s1, b, a)
y_s2 = ecuacion_diferencias(s2, b, a)
y_s3 = ecuacion_diferencias(s3, b, a)
y_sq = ecuacion_diferencias(sq, b, a)

# Calcular energía y potencia
for y, name in zip([y_s1, y_s2, y_s3, y_sq], ['s1', 's2', 's3', 'sq']):
    E, P = energia_potencia(y)
    print(f"{name}: Energía = {E:.4f}, Potencia = {P:.4f}")

# Graficar salidas
plot_signal(n, y_s1, "Salida con entrada s1[n]")
plot_signal(n, y_s2, "Salida con entrada s2[n]")
plot_signal(n, y_s3, "Salida con entrada s3[n]")
plot_signal(n, y_sq, "Salida con entrada sq[n]")

# Comparación convolución
ysc1_np = np.convolve(s1, h)[:N]
plt.figure()
plt.plot(n, y_s1, marker='o', linestyle='-', label='Ecuación diferencias')
plt.plot(n, ysc1_np, marker='x', linestyle='--', label='Convolución')
plt.title("Comparación salida s1[n]")
plt.xlabel("desplazamiento [n]")
plt.ylabel("y[n]")
plt.grid(True)
plt.legend()
plt.show()

# ----------------------
# Ejemplos con coeficientes distintos
# ----------------------
b = [0.0]*11
b[0]  = 1.0
b[10] = 3.0
a = [0.0]*11
y_s1 = ecuacion_diferencias(delta, b, a)
plot_signal(n, y_s1, "Salida con coeficientes variante 1 con δ[n]=x[n].")

b[0] = 1.0
a[10]= 3.0
y__s1 = ecuacion_diferencias(delta, b, a)
plot_signal(n, y__s1, "Salida con coeficientes variante 2 con δ[n]=x[n]")

s1_ = np.convolve(s1, y_s1)[:N] # SEÑAL CON COEFICIENTES 1 Y 3 EN EL VECTOR b
s1__ = np.convolve(s1, y__s1)[:N] # SEÑAL CON COEFICIENTES 1 EN B y 3 EN a
plot_signal(n, s1_, "Salida con coeficientes variante 1 con s1[n]=x[n]")
plot_signal(n, s1__, "Salida con coeficientes variante 2 con s1[n]=x[n]")




