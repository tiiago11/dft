import numpy as np

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

def calculate_magnitude_2d(x):
    x = np.asarray(x, dtype=complex)
    M = len(x)
    N = len(x[0])
    mat = np.zeros((M, N), np.float)
    for i in range(M):
        for j in range(N):
            mat[i][j] = np.sqrt(x[i][j].real*x[i][j].real+x[i][j].imag*x[i][j].imag)

    return mat

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

