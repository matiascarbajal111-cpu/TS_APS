# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:06:20 2025

@author: Matias
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

def mi_funcion_sin(vmax=1.0, dc=0.0, ff=1.0, ph=0.0, nn=1000, fs=1000.0, como_columna=True):
    tt = np.arange(nn, dtype=float) /(fs)
    xx = dc + vmax * np.sin(2 * np.pi * ff * tt + ph)
    return tt, xx

N=1000
fs=N
df=fs/N
ts=1/fs

amplitud = 1
#-----------------------------------------------------------
k0 = N/4                 
tt_0,xx_0=mi_funcion_sin(vmax=amplitud, dc=0, ff=k0*df, ph=0, nn=N, fs=fs)##SEÑAL 0
X0 = fft(xx_0)
#-----------------------------------------------------------
k0 = (N/4) + 0.25
tt_1,xx_1 = mi_funcion_sin(vmax=amplitud, dc=0, ff=k0*df, ph=0, nn=N, fs=fs)##SEÑAL 1
#-----------------------------------------------------------
k0 = (N/4) + 0.5
tt_2,xx_2 = mi_funcion_sin(vmax=amplitud, dc=0, ff=k0*df, ph=0, nn=N, fs=fs)##SEÑAL 2
#-----------------------------------------------------------


X0 = fft(xx_0)
X0abs = np.abs(X0)
X0ang = np.angle(X0)

X1 = fft(xx_1)
X1abs = np.abs(X1)
X1ang = np.angle(X1)

X2 = fft(xx_2)
X2abs = np.abs(X2)
X2ang = np.angle(X2)



frec = np.arange(N)*df


plt.figure(1)
plt.plot(frec,np.log10(X0abs)*20,'x',label= 'X0 abs en dB N/4 ')
plt.plot(frec,np.log10(X1abs)*20,'x',label= 'X1 abs en dB N/4 + 0.25')
plt.plot(frec,np.log10(X2abs)*20,'o',label= 'X2 abs en dB (N/4 + 0.5')
#plt.plot(frec,X0abs,'x',label= 'X0 abs N/4')
#plt.plot(frec,X1abs,'o',label= 'X1 abs N/4 + 0.25')
#plt.plot(frec,X2abs,'o',label= 'X2 abs N/4 + 0.5')

plt.xlim([0,fs/2])
#######
plt.title('FFT')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.legend()
plt.grid()
plt.show()

#fig, axes = plt.subplots(nrows=4, ncols=1)
#axes[0].plot(t1, x1)
#axes[1].plot(t1, X1, 'x')
#axes[2].plot(t2, x2, 'x')
#axes[3].plot(t2, X2, 'x')
#plt.grid()
#plt.show()