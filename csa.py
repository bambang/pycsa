#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

# extended chirp scaling algorithm 
# system parameters:
c = float(299792458)  # speed of light in vacuum
f_c = float(5331004416)  # carrier frequency
f_s = float(19207680)  # data sampling rate
echoes = float(2206)  # number of radar echoes in data file
samples = float(672)  # number of samples per radar echo
t_near = 0.00533035424126647  # near range fast time (2 x range)

tau = np.arange(t_near, t_near+ samples/f_s-1/f_s , 1/f_s) # fast time
f_dc = 0  # Doppler centroid (squint angle)
v = 7500  # SAR satellite platform velocity
PRF = 2067.120103315  # pulse repetition frequency
t_p = 2.160594095e-005  # chirp pulse duration
B = 16000000  # chirp bandwidth

# additional parameters, see paper
r_ref = (tau[0]+samples/2/f_s)/2*c
alpha = 1  
f_a = np.arange(-PRF/2+f_dc,f_dc+PRF/2-PRF/echoes,PRF/echoes)
f_r = np.arange(-f_s/2, f_s/2-f_s/samples, f_s/samples)
lbd = c/f_c

# example raw data point target from file 'pointtarget.raw'
# the file contains a simulated point target with Envisat ASAR
# read data from file 
filename = 'pointtarget.raw' 

# desired region 
firstEcho=1 
numEchoes=2206        

f = file(filename,'rb')   # read in binary mode, must do on windows!
data_type_int32 = np.dtype('int32').newbyteorder('B') # big endian 
data_type_float32 = np.dtype('float32').newbyteorder('B') # big endian 
data_type_complex64 = np.dtype('complex64').newbyteorder('B') # big endian 
headerInts = np.fromfile(f, dtype=data_type_int32,count=8)

#ras_magic=headerInts[0]
ras_width=headerInts[1]
#ras_height=headerInts[2]
#ras_depth=headerInts[3]
#ras_length=headerInts[4]
#ras_type=headerInts[5]
#ras_maptype=headerInts[6]
#ras_maplength=headerInts[7]

# skip to desired line 
f.seek(data_type_complex64.itemsize*ras_width*(firstEcho-1),1)

#read_count=0    
echoData = np.fromfile(f, dtype=data_type_complex64, count=numEchoes*ras_width).reshape(numEchoes,ras_width)
#print curEcho.size
f.close()    

# plot the raw echo data
plt.figure(1)
plt.subplot(121)
plt.imshow(np.angle(echoData))
plt.gray()
plt.title('Raw Data (real)')

# azimuth fft 
data = np.fft.fftshift(np.fft.fft(np.fft.fftshift(echoData, axes=0),axis=0),axes=0)

# display 
plt.subplot(122)
plt.title('After Azimuth FFT (real)')
plt.imshow(np.angle(data))

plt.show()

# ========================================================================
# chirp scaling, range scaling: H1
# ========================================================================
beta = (1 - (f_a*lbd/2/v)**2)**0.5
a = 1/beta - 1
R = r_ref/beta
a_scl = a + ((1-alpha)*(1+a))/alpha
k_r = -B/t_p

# WARNING: in the work of Moreira et al, 1996 k_r is defined negative
# because they assume a "down-chirp", whereas we assume an "up-chirp" and
# must explicitely set k_r as negative for the following transfer functions
# to be equal to the theoretical formulations.

k_inv = 1/k_r - (2*lbd *r_ref *(beta**2-1))  /(c**2* (beta**3))
k = 1/k_inv

x = k*a_scl
bigx = x*np.ones((1,samples))
Tau = np.ones(echoes,1)*tau
y = 2.*R./c
Y = y(:)*ones(1,samples);
Z = (Tau-Y).^2;
H1 = exp(-j*pi*X.*Z);

data = data .* H1;