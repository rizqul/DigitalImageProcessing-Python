# edge.py
# JM Kinser
# Python 3.4 +

import numpy as np
import scipy.ndimage as nd
import scipy.signal as ss

# create a 512x512 image with several blocks of increasing fuzziness
def FuzzBlocks():
    data = np.zeros((512,512))
    k = 0.0
    for v in range( 7 ):
        for h in range( 7 ):
            # create the image
            temp = np.zeros((64,64))
            temp[16:48,16:48] = 1
            if k > 0:
                temp = cspline2d( temp, k )
            # place in large array
            vv = (v+1)*64-32
            hh = (h+1)*64-32
            data[vv:vv+64,hh:hh+64] = temp + 0
            k += 1
    return data

# simple vertical edge enhancement
def VertEdge( indata, m=1 ):
    b = nd.shift( indata, (0,m))
    answ = abs( indata - b )
    return answ

def DerivEdge( indata, dvh ):
    b = nd.shift( indata + 0.0, dvh, order=0, cval=0 ) 
    answ = abs(indata - b)
    return answ

def Sobel( indata ):
    gh = nd.sobel( indata +0., axis=0)
    gv = nd.sobel( indata+0., axis=1)
    edj = abs(gh) + abs(gv)
    return edj

def DoGFilter( amg, sigma1, sigma2 ):
    b1 = nd.gaussian_filter( amg, sigma1 )
    b2 = nd.gaussian_filter( amg, sigma2 )
    answ = b1 - b2
    return answ

def Harris( adata, alpha=0.2 ):
    Ix = nd.sobel( indata, 0 )
    Iy = nd.sobel( indata, 1 )
    Ix2 = Ix**2;     Iy2 = Iy**2
    Ixy = abs(Ix * Iy)
    Ix2 = nd.gaussian_filter( Ix2, 3 )
    Iy2 = nd.gaussian_filter( Iy2, 3 )
    Ixy = nd.gaussian_filter( Ixy, 3 )
    detC = Ix2 * Iy2 - 2 * Ixy
    trC = Ix2 + Iy2
    R = detC - alpha * trC**2
    return R
