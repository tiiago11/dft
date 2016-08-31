import wave
import numpy as np
from pylab import *
import FourierTransform as ourft



######################################################
############################ Test ours against numpy's
######################################################
w = wave.open("./audio-samples/lz-s2h-solo.wav", "rb")
frames = w.readframes(2048)
array = np.frombuffer(frames, dtype = "ubyte")

ours_fff = ourft.DFT_slow(array)
ours_iff = ourft.IDFT_slow(ours_fff)

numpys_fff = np.fft.fft(array)
numpys_iff = np.fft.ift(numpys_fff)

print(np.allclose(ours_fff, numpys_fff)) # verify our dft against known fft implementation
print(np.allclose(ours_iff, numpys_iff)) # verify our idft against known ifft implementation





#######################################################
##################################### Truncate and plot
#######################################################
#truncabove = True
#trunc_threshold = 2048 * 14

#w = wave.open("./audio-samples/lz-s2h-solo.wav", "rb")
#frames = w.readframes(2048)
#array = np.frombuffer(frames, dtype = "ubyte")
#fff = np.fft.fft(array)
## truncate
#if (truncabove):
#    fff = ourft.zero_upper_range(fff, trunc_threshold)
#else:
#    fff = ourft.zero_lower_range(fff, trunc_threshold)

#iff = np.fft.ifft(fff)

## plot
#N = len(iff)
#t = np.arange(0, N)
#plot(t, iff, label='reconstructed')
#plot(t, array, label='original')
#legend(loc='upper right')
#axis([0, N, amin(array), amax(array)]) #[minx, maxx, miny,maxy]
#xlabel('sample')
#ylabel('amplitude')

#if (truncabove):
#    title('Truncated DFT above ' + str(trunc_threshold))
#else:
#    title('Truncated DFT below ' + str(trunc_threshold))
#show()





######################################################
######################### Truncate and save audio file
######################################################

#w = wave.open("./audio-samples/lz-s2h-solo.wav", "rb")
#rec = wave.open("./output/reconstructed.wav", "wb")
#rec.setparams(w.getparams())

#framesread = 0
#while framesread < 2048 * 200:
#    framesread += 2048
#    frames = w.readframes(2048)
#    array = np.frombuffer(frames, dtype = "ubyte")
#    fff = np.fft.fft(array)
#    fff = ourft.zero_upper_range(fff, 2048 * 14)
#    #fff = ourft.zero_lower_range(fff, -2048 * 14)
#    iff = np.fft.ifft(fff)
#    rec.writeframes(np.asarray(iff, dtype=int8))

#rec.close()

