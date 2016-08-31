import numpy as np
from PIL import Image
from pylab import *
import FourierTransform as ourft


######################################################
############################ Test ours against numpy's
######################################################
#img = Image.open("images/pulse_512_diag.png").convert("L") # use greyscale
#pixels = np.asarray(img.getdata(), dtype=np.float32).reshape((img.size[1],img.size[0]))
#(M,N) = img.size

#print("Computing numpy's direct...")
#numpys_ff2d = np.fft.fft2(pixels)
#print("Computing numpy's inverse...")
#numpys_iff2d = np.fft.ifft2(numpys_ff2d)

#print("Computing ours direct...")
#ours_ff2d = ourft.DFT2D_slow(pixels)
#print("Computing ours inverse...")
#ours_iff2d = ourft.IDFT2D_slow(ours_ff2d)

#print("Is our FT correct? " + str(np.allclose(ours_ff2d, numpys_ff2d)))
#print("Is our IFT correct? " + str(np.allclose(ours_iff2d, numpys_iff2d)))



######################################################
######## Compute FT, IFT, Mag, Phase angle and plot
######################################################
#img = Image.open("images/pulse_512_diag.png").convert("L") # use greyscale
#pixels = np.asarray(img.getdata(), dtype=np.float32).reshape((img.size[1],img.size[0]))
#(M,N) = img.size

## compute forward 
#print("Computing direct...")
#ff2d = np.fft.fft2(pixels)

## compute inverse 
#print("Computing inverse...")
#iff2d = np.fft.ifft2(ff2d)

## compute mag map
## adjust frequencies so the center of the spectrum is at the center of the image.
#mag_pixels = np.ndarray((M,N))
#for x in range(M):
#    for y in range(N):        
#        mag_pixels[x,y] = pixels[x,y] * np.power(-1, x+y) # according to gonzales, the -1^(x+y) is to center the spectrum

#print("Computing direct (magnitude)...")
#mag_ff2d = np.fft.fft2(mag_pixels)
#magMap = ourft.calculate_magnitude_2d(mag_ff2d)
#for x in range(M):
#    for y in range(N):
#        mag_pixels[x,y] = 15 * int(np.log(magMap[y,x])) #magnitude highlight: 

#phase_map = ourft.calculate_phase_angle_2d(mag_ff2d)

## plot
#f = plt.figure()

#f.add_subplot(2, 2, 1)
#title('Original')
#plt.imshow(pixels, interpolation='nearest', cmap='gray')

#f.add_subplot(2, 2, 2)
#title('Reconstructed')
#plt.imshow(np.asarray(iff2d, dtype=float), interpolation='nearest', cmap='gray')

#f.add_subplot(2, 2, 3)
#title('Magnitude/Spectrum/Frequency')
#plt.imshow(mag_pixels, interpolation='nearest', cmap='gray')

#f.add_subplot(2, 2, 4)
#title('Phase angle')
#plt.imshow(phase_map, interpolation='nearest', cmap='gray')

#plt.show()




######################################################
######## Compute FT, IFT, Mag, Phase angle and plot
######################################################
img = Image.open("images/lena.jpg").convert("L") # use greyscale
pixels = np.asarray(img.getdata(), dtype=np.float32).reshape((img.size[1],img.size[0]))
(M,N) = img.size

# compute forward 
print("Computing direct...")
ff2d = np.fft.fft2(pixels)




real_zeroed_ff2d = np.zeros((M, N), np.complex)
for i in range(M):
    for j in range(N):
        real_zeroed_ff2d[i][j] = np.complex(0, ff2d[i][j].imag)

# compute inverse 
print("Computing inverse...")
real_zeroed_iff2d = np.fft.ifft2(real_zeroed_ff2d)





imag_zeroed_ff2d = np.zeros((M, N), np.complex)
for i in range(M):
    for j in range(N):
        imag_zeroed_ff2d[i][j] = np.complex(ff2d[i][j].real, 0)

# compute inverse 
print("Computing inverse...")
imag_zeroed_iff2d = np.fft.ifft2(imag_zeroed_ff2d)





# plot
f = plt.figure()

f.add_subplot(2, 2, 1)
title('Original')
plt.imshow(pixels, interpolation='nearest', cmap='gray')

f.add_subplot(2, 2, 2)
title('Reconstructed: zeroed real')
plt.imshow(np.asarray(real_zeroed_iff2d, dtype=float), interpolation='nearest', cmap='gray')

f.add_subplot(2, 2, 3)
title('Reconstructed: zeroed imag')
plt.imshow(np.asarray(imag_zeroed_iff2d, dtype=float), interpolation='nearest', cmap='gray')

plt.show()