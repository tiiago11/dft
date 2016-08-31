import numpy as np
import wave
import os
from pylab import *
import matplotlib.pyplot as plt
from PIL import Image

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x
    from https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/"""
    x = np.asarray(x, dtype=complex)
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def IDFT_slow(x):
    """Compute the inverse discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=complex)
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
    return 1.0/N * np.dot(M, x)

def DFT2D_slow(x):
    """Compute the discrete Fourier Transform of the 2D array x"""
    x = np.asarray(x, dtype=complex)
    
    M = len(x)
    N = len(x[0])
    mat = np.zeros((M, N), np.complex)
    
    for row in range(M):
        mat[row] = DFT_slow(x[row])

    for col in range(N):
        mat[:,col] = DFT_slow(mat[:,col])

    return mat

def IDFT2D_slow(x):
    """Compute the discrete Fourier Transform of the 2D array x"""
    x = np.asarray(x, dtype=complex)
    
    M = len(x)
    N = len(x[0])
    mat = np.zeros((M, N), np.complex)
    
    for row in range(M):
        mat[row] = IDFT_slow(x[row])

    for col in range(N):
        mat[:,col] = IDFT_slow(mat[:,col])

    return mat

#def DFT2D_slow(x):
#    """Compute the discrete Fourier Transform of the 2D array x
#       TODO fix"""
#    x = np.asarray(x, dtype=complex)
#    M = len(x)
#    N = len(x[0])
#    mat = np.zeros((M, N), np.complex)
    
#    sumOutter = 0
#    sumInner = 0
#    for k in range(M):
#        for l in range(N):
#            for m in range(M):
#                for n in range(N):
#                    sumInner += x[m][n] * np.exp(-2j * np.pi * ( float(m*k)/float(M) + float(n*l)/float(N)))
#                sumOutter += sumInner
#            mat[l][k] = sumOutter / float(M*N)
#            sumOutter = 0
#            sumInner = 0

#    return mat;

def zero_upper_range(x, upper_threshold):
    """set to zero values larger than upper_threshold"""
    x = np.asarray(x, dtype=complex)
    count = 0
    for i in range(0, x.shape[0], 1):
        if x[i].real > upper_threshold:
            x[i] = complex(0, 0j)
            count+=1
    print("zeroed samples: ", count)
    return x;

def zero_lower_range(x, lower_threshold):
    """set to zero values lower than lower_threshold"""
    x = np.asarray(x, dtype=complex)
    count = 0
    for i in range(0, x.shape[0], 1):
        if x[i].real < lower_threshold:
            x[i] = complex(0, 0j)
            count+=1
    print("zeroed samples: ", count)
    return x;

######################################################
########################## test 1D DFT with audio wave
######################################################

##load audio
#w = wave.open("./audio-samples/lz-s2h-beginning.wav", "rb")
#print('Frames: ' + str(w.getnframes()))
#print('Channels: ' + str(w.getnchannels()))
#frames = w.readframes(512)
#array = np.frombuffer(frames, dtype = "ubyte")

##test dft
#fff = DFT_slow(array)
#iff = IDFT_slow(fff)
#print(np.allclose(fff, np.fft.fft(array))) # verify our dft against known fft implementation
#print(np.allclose(IDFT_slow(fff), np.fft.ifft(fff))) # verify our idft against known ifft implementation

##truncation
#print("input samples: ", len(array))
##ff_truncated = zero_upper_range(fff, 200) # set to zero values larger than
#ff_truncated = zero_lower_range(fff, 10)   # set to zero values lower than
#iff_truncated = IDFT_slow(ff_truncated) # reconstruct the signal

##gen graph
#N = len(iff)
#t = arange(0, N)
#plot(t, iff_truncated)
#plot(t, array)
#axis([0, N, amin(array), amax(array)]) #[minx, maxx, miny,maxy]
#xlabel('sample')
#ylabel('amplitude')
#title('Stairway To Heaven DFT')

#show() # plot audio wave

###################################################
########################## Test 2D DFT with image
###################################################

#load image
img = Image.open("images/lena_128.png")
rgb_im = img.convert('RGB')
pixels = rgb_im.load()
(W,H) = rgb_im.size
red_channel = np.zeros((W, H), np.float) # for now, we want only the red channel
for x in range(H):
    for y in range(W):
        r, g, b = pixels[x,y]
        red_channel[x][y] = r

#compute forward 
ff2d = DFT2D_slow(red_channel)
print(np.allclose(ff2d, np.fft.fft2(red_channel)))
#compute inverse 
iff2d = IDFT2D_slow(ff2d)
print(np.allclose(iff2d, np.fft.ifft2(ff2d)))
#compare original signal to the inverse
print(np.allclose(iff2d, red_channel))

#save red-ish image
image = Image.new("RGB", (W, H))
img_data = image.load() 
for x in range(H):
    for y in range(W):
        img_data[x,y] = (int(iff2d[x][y].real), 0, 0)

image.save("output.png", "PNG")

