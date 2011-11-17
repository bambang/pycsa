# read binary data from sun raster file
# result is matrix  [ numEchoes , elmcount ] of complex number
import numpy as np
import matplotlib.pyplot as plt

f = file('pointtarget.raw','rb')   # read in binary mode, must do on windows!
data_type_int32 = np.dtype('int32').newbyteorder('B') # big endian 
data_type_float32 = np.dtype('float32').newbyteorder('B') # big endian 
data_type_complex64 = np.dtype('complex64').newbyteorder('B') # big endian 
headerInts = np.fromfile(f, dtype=data_type_int32,count=8)

ras_magic=headerInts[0]
ras_width=headerInts[1]
ras_height=headerInts[2]
ras_depth=headerInts[3]
ras_length=headerInts[4]
ras_type=headerInts[5]
ras_maptype=headerInts[6]
ras_maplength=headerInts[7]


# skip to desired line 
firstEcho=1 
#for k in range(1,firstEcho):  # from 1 to firstEcho-1 
#skippedEcho = np.fromfile(f, dtype=data_type_complex64, count=ras_width*(firstEcho-1))
f.seek(data_type_complex64.itemsize*ras_width*(firstEcho-1),1)

numEchoes=2206        
#read_count=0    
elmcount  = ras_width * 2
#for k in range(1,numEchoes+1):
#echoData = np.fromfile(f, dtype=data_type_float32, count=numEchoes*elmcount).reshape(numEchoes,elmcount)
echoData = np.fromfile(f, dtype=data_type_complex64, count=numEchoes*ras_width).reshape(numEchoes,ras_width)
#print curEcho.size
f.close()    
plt.imshow(np.abs(echoData))
plt.show()