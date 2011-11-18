import numpy as np
import matplotlib.pyplot as plt


def readdata:
    
    return
# extended chirp scaling
# example raw data point target from file 'pointtarget.raw'
# the file contains a simulated point target with Envisat ASAR
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

#
