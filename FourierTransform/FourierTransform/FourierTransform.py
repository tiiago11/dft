import numpy as np
import wave
import os

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

#load audio
w = wave.open("./audio-samples/dogs-n-birds.wav", "rb")
print('Frames: ' + str(w.getnframes()))
print('Channels: ' + str(w.getnchannels()))
frames = w.readframes(8)
array = np.frombuffer(frames, dtype = "ubyte")

#for x in array:
#    print(str(x))

print(DFT_slow(array))