
#
#
#  Header structure:
#  datatype = 0l
#  gridtype = 0l
#  sizeG    = 0l
#  sizeX    = 0l
#  sizeY    = 0l
#  sizeZ    = 0l
#  offsX    = float(0)
#  offsY    = float(0)
#  offsZ    = float(0)
#  box      = fltarr(3)
#  remaining_bytes = bytarr(128-36-12)
#  header_high     = bytarr(128)

#  Data types:
#  0        bit                 1/8       1                0 -> 1
#  1        signed char          1        8             -128 -> +127
#  2        unsigned char        1        8                0 -> +255
#  3        short int            2       16          -32,768 -> +32,767
#  4        unsigned short int   2       16                0 -> +65,535
#  5        int                  4       32   -2,147,483,648 -> +2,147,483,647
#  6        unsigned int         4       32                0 -> +4,294,967,295
#  7        long int             4       32   -2,147,483,648 -> +2,147,483,647
#  8        float                4       32       3.4 E-38   -> 3.4 E+38
#  9        double               8       64       1.7 E-308  -> 1.7 E+308
# 10        long double          12      96       3.4 E-4932 -> 3.4 E+4932
#
#
#

import struct
import numpy as np
import array as arr

#---------------------------------------------------------
#
#---------------------------------------------------------
def project_volu(vol, lim1, lim2, ax):
    shp = vol.shape
    out = np.zeros(shape=(shp[0], shp[1]))

    for i in range(lim1,lim2):
        if (ax == 0):
            out = out + vol[i,:,:]
        if (ax == 1):
            out = out + vol[:,i,:]
        if (ax == 2):
            out = out + vol[:,:,i]
    return out


#---------------------------------------------------------
#
#---------------------------------------------------------
def rotate_cube(vol, ax, new=False):
    shp = vol.shape

    if (ax == 0):
        for i in range(shp[0]):
            vol[i,:,:] = np.rot90(vol[i,:,:])
    if (ax == 1):
        for i in range(shp[1]):
            vol[:,i,:] = np.rot90(vol[:,i,:])
    if (ax == 2):
        for i in range(shp[2]):
            vol[:,:,i] = np.rot90(vol[:,:,i])
    return vol


#================================================================================
#
#================================================================================


#-----------------------------------
#
#-----------------------------------
def read_volume_header(filename):

    F = open(filename,'rb')

    #--- Read header
    head = F.read(256)
    (sizeX,) = struct.unpack('i',head[12:16])
    (sizeY,) = struct.unpack('i',head[16:20])
    (sizeZ,) = struct.unpack('i',head[20:24])

    return [sizeX,sizeY,sizeZ]


#-----------------------------------
#
#-----------------------------------
def read_dvolume(filename):

    F = open(filename,'rb')

    #--- Read header
    head = F.read(256)
    (sizeX,) = struct.unpack('i',head[12:16])
    (sizeY,) = struct.unpack('i',head[16:20])
    (sizeZ,) = struct.unpack('i',head[20:24])
    print('>>> Reading volume of size:', sizeX,sizeY,sizeZ)

    den = arr.array('d')
    den.fromfile(F,sizeX*sizeY*sizeZ)
    F.close()
    den = np.array(den).reshape((sizeX,sizeY,sizeZ)).astype(np.float64)

    return den


#-----------------------------------
#
#-----------------------------------
def read_fvolume(filename):

    F = open(filename,'rb')

    #--- Read header
    head = F.read(256)
    (sizeX,) = struct.unpack('i',head[12:16])
    (sizeY,) = struct.unpack('i',head[16:20])
    (sizeZ,) = struct.unpack('i',head[20:24])
    print('>>> Reading volume of size:', sizeX,sizeY,sizeZ)

    den = arr.array('f')
    den.fromfile(F,sizeX*sizeY*sizeZ)
    F.close()
    den = np.array(den).reshape((sizeX,sizeY,sizeZ)).astype(np.float32)

    return den

#-----------------------------------
#
#-----------------------------------
def read_ivolume(filename):

    F = open(filename,'rb')

    #--- Read header
    head = F.read(256)
    (sizeX,) = struct.unpack('i',head[12:16])
    (sizeY,) = struct.unpack('i',head[16:20])
    (sizeZ,) = struct.unpack('i',head[20:24])
    print('>>> Reading volume of size:', sizeX,sizeY,sizeZ)

    den = arr.array('b')
    den.fromfile(F,sizeX*sizeY*sizeZ)
    F.close()
    den = np.array(den).reshape((sizeX,sizeY,sizeZ)).astype(np.int8)

    return den

#-----------------------------------
#
#-----------------------------------
def read_bvolume(filename):

    F = open(filename,'rb')

    #--- Read header
    head = F.read(256)
    (sizeX,) = struct.unpack('i',head[12:16])
    (sizeY,) = struct.unpack('i',head[16:20])
    (sizeZ,) = struct.unpack('i',head[20:24])
    print('>>> Reading volume of size:', sizeX,sizeY,sizeZ)

    den = arr.array('b')
    den.fromfile(F,sizeX*sizeY*sizeZ)
    F.close()
    den = np.array(den).reshape((sizeX,sizeY,sizeZ)).astype(np.uint8)

    return den


#-----------------------------------
#
#-----------------------------------
def write_fvolume(vol, filename):

    shp = vol.shape

    #--- Define header
    datatype = 8
    gridtype = 0
    sizeG    = shp[0]
    sizeX    = shp[0]
    sizeY    = shp[1]
    sizeZ    = shp[2]
    offX     = 0.0
    offY     = 0.0
    offZ     = 0.0
    box      = 1.0
    h0 = np.array([datatype, gridtype, sizeG, sizeX,sizeY,sizeZ], dtype='int32')
    h1 = np.array([offX,offY,offZ], dtype='float32')
    h2 = np.array([box,box,box], dtype='float32')
    h3 = np.zeros(208,dtype='uint8')

    #--- Binary write
    F = open(filename, "bw")

    #--- Write header to file
    h0.tofile(F)
    h1.tofile(F)
    h2.tofile(F)
    h3.tofile(F)

    #--- write volume data
    vol.astype(dtype='float32').tofile(F)

    F.close()


#-----------------------------------
#
#-----------------------------------
def write_bvolume(vol, filename):

    shp = vol.shape

    #--- Define header
    datatype = 8
    gridtype = 0
    sizeG    = shp[0]
    sizeX    = shp[0]
    sizeY    = shp[1]
    sizeZ    = shp[2]
    offX     = 0.0
    offY     = 0.0
    offZ     = 0.0
    box      = 1.0
    h0 = np.array([datatype, gridtype, sizeG, sizeX,sizeY,sizeZ], dtype='int32')
    h1 = np.array([offX,offY,offZ], dtype='float32')
    h2 = np.array([box,box,box], dtype='float32')
    h3 = np.zeros(208,dtype='uint8')

    #--- Binary write
    F = open(filename, "bw")

    #--- Write header to file
    h0.tofile(F)
    h1.tofile(F)
    h2.tofile(F)
    h3.tofile(F)

    #--- write volume data
    vol.astype(dtype='uint8').tofile(F)

    F.close()
