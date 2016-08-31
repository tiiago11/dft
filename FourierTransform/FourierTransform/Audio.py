######################################################
########################## test 1D DFT with audio wave
######################################################
import wave
import numpy as np
from pylab import *
import FourierTransform as ourft

#load audio
w = wave.open("./audio-samples/lz-s2h-solo.wav", "rb")
#w = wave.open("./audio-samples/bass.wav", "rb")
print('Frames: ' + str(w.getnframes()))
print('Channels: ' + str(w.getnchannels()))
frames = w.readframes(64)
array = np.frombuffer(frames, dtype = "ubyte")

#test dft
fff = ourft.DFT_slow(array)
iff = ourft.IDFT_slow(fff)
print(np.allclose(fff, np.fft.fft(array))) # verify our dft against known fft implementation
print(np.allclose(ourft.IDFT_slow(fff), np.fft.ifft(fff))) # verify our idft against known ifft implementation

#truncation
print("input samples: ", len(array))
#ff_truncated = ourft.zero_upper_range(fff, 200) # set to zero values larger than
truncatebelow = 10
ff_truncated = ourft.zero_lower_range(fff, truncatebelow)   # set to zero values lower than
iff_truncated = ourft.IDFT_slow(ff_truncated) # reconstruct the signal

#gen graph
N = len(iff)
t = np.arange(0, N)
plot(t, iff_truncated, label='reconstructed')
plot(t, array, label='original')
legend(loc='upper right')
axis([0, N, amin(array), amax(array)]) #[minx, maxx, miny,maxy]
xlabel('sample')
ylabel('amplitude')
title('Truncated DFT below ' + str(truncatebelow))

# plot audio wave
show()
