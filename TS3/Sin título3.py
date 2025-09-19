import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft

def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    """
    Genera una senoide:
      vmax: amplitud
      dc: offset en DC
      ff: frecuencia (Hz)
      ph: fase (radianes)
      nn: número de muestras
      fs: frecuencia de muestreo (Hz)
    Devuelve (tt, xx)
    """
    Ts = 1 / fs
    tt = np.linspace(0, (nn - 1) * Ts, nn)
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc
    return tt, xx

# ------------------------------------------------
N = 1000  # cuanto mas chico N, mas separados se ven los "palitos" del espectro
fs = N   # frecuencia de muestreo (en las imágenes fs = N)
df = fs / N  # resolución espectral

# en una FFT, el eje x no son muestras de tiempo sino índices de frecuencia
# como gráfico de 0 a N/2, mis frecuencias quedan de fs/2
freqs = np.arange(0, N) * df
plt.figure(1)

# ------------------- N/4 -------------------------
ff = (N / 4) * df
_, yy = mi_funcion_sen(1, 0, ff, 0, N, fs)
FFT = fft(yy)
absFFT = np.abs(FFT)
plt.stem(freqs, absFFT, linefmt="orchid", markerfmt="o", basefmt="orchid",
         label=f"ff = {ff:.1f} Hz (N/4)")

# --------------- (N/4 + 1)*df --------------------
ff1 = (N / 4 + 1) * df
_, yy1 = mi_funcion_sen(1, 0, ff1, 0, N, fs)
FFT1 = fft(yy1)
absFFT1 = np.abs(FFT1)
plt.stem(freqs, absFFT1, linefmt="lightseagreen", markerfmt="o", basefmt="lightseagreen",
         label=f"ff1 = {ff1:.1f} Hz (N/4 + 1)")

# ------------------ del medio --------------------
ff2 = (ff + ff1) / 2
_, yy2 = mi_funcion_sen(1, 0, ff2, 0, N, fs)
FFT2 = fft(yy2)
absFFT2 = np.abs(FFT2)
plt.stem(freqs, absFFT2, linefmt="deepskyblue", markerfmt="o", basefmt="deepskyblue",
         label=f"ff2 = {ff2:.1f} Hz (medio)")

# ------------------ formato del plot ----------------
plt.title("FFT de senoidales")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X[k]|")
plt.grid(True)
plt.xlim(0, N / 2)
plt.legend()
# (el código original se detenía en plt.legend())
