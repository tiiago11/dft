import numpy as np
import wave
import os
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

#load audio
w = wave.open("./audio-samples/lz-s2h-beginning.wav", "rb")
print('Frames: ' + str(w.getnframes()))
print('Channels: ' + str(w.getnchannels()))
frames = w.readframes(256)
array = np.frombuffer(frames, dtype = "ubyte")

#test dft
fff = DFT_slow(array)
iff = IDFT_slow(fff)
print(np.allclose(fff, np.fft.fft(array))) # verify our dft against known fft implementation
print(np.allclose(IDFT_slow(fff), np.fft.ifft(fff))) # verify our idft against known ifft implementation

#get the real part
iff_real = []
for i in range(0, iff.shape[0], 1):
    iff_real.append(iff[i].real * 0.9) # Scale a little bit to look different

#gen graph
N = len(iff_real)
t = arange(0, N)
plot(t, iff_real)
plot(t, array)
axis([0, N, amin(array)*10, amax(array)]) #[minx, maxx, miny,maxy]
xlabel('sample')
ylabel('amplitude')
title('Stairway To Heaven DFT')

show()