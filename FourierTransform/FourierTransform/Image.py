##################################################
######################### Test 2D DFT with image
##################################################
import numpy as np
from PIL import Image
import FourierTransform as ourft

#load image
img = Image.open("images/lena_128.png").convert("L") # use greyscale
pixels = np.asarray(img.getdata(), dtype=np.float32).reshape((img.size[1],img.size[0]))
(M,N) = img.size

#adjust frequencies so the center of the spectrum is at the center of the image.
for x in range(M):
    for y in range(N):        
        pixels[x,y] *= np.power(-1, x+y) # according to gonzales, the -1^(x+y) is to center the spectrum

##########
#compute forward 
print("Calculating direct...")
ff2d = ourft.DFT2D_slow(pixels)
#print(np.allclose(ff2d, np.fft.fft2(pixels)))

##########
#compute inverse 
print("Calculating inverse...")
iff2d = ourft.IDFT2D_slow(ff2d)
#print(np.allclose(iff2d, np.fft.ifft2(ff2d)))

##########
#compare original signal to the inverse
#print(np.allclose(iff2d, pixels))

###########
#### save reconstructed image
#image = Image.new("L", (M, N))
#img_data = image.load()
#for x in range(M):
#    for y in range(N):
#        img_data[x,y] = int(iff2d[y,x].real)
#image.save("output.png", "PNG")

##########
## Save magnitude spectrum image
magMap = ourft.calculate_magnitude_2d(ff2d)
image = Image.new("L", (M, N))
img_data = image.load() 
for x in range(M):
    for y in range(N):
        img_data[x,y] = 15 * int(np.log(magMap[y,x])) #magnitude highlight: 
image.save("output/output_mag.png", "PNG")