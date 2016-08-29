import numpy as np
from pylab import *
import matplotlib.pyplot as plt

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x
    from https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def IDFT_slow(x):
    """Compute the inverse discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    return 1.0/N * np.dot(M, x)

#np.random.seed(1)
x = np.random.random(1024)
#print(x)
fff = DFT_slow(x)
iff = IDFT_slow(fff)
#print(fff)
#print(iff)
#print(np.fft.ifft(DFT_slow(x)))
print(np.allclose(fff, np.fft.fft(x)))
print(np.allclose(IDFT_slow(fff), np.fft.ifft(fff)))

N = x.shape[0]
t = arange(0, N)
plot(t, x)
axis([0, N, amin(x), amax(x)]) #[minx, maxx, miny,maxy]
xlabel('time (s)')
ylabel('current (nA)')
title('Gaussian colored noise')

show()
