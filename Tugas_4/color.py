# color.py
# (c) Copyright JM Kinser 2018
# The Python scripts contained in this file can only be used for educational purposes. All other rights are reserved by the author.
import numpy as np

# color code: blue to red
def ColorCode1( mglist ):
    N = len( mglist ) # number of images
    V,H = mglist[0].shape
    answ = np.zeros( (V,H,3) )
    for i in range( N ):
        scale = 256./N * i
        blue = 256. - scale
        answ[:,:,0] += mglist[i] * scale
        answ[:,:,2] += mglist[i] * blue
    mx = answ.max()
    answ /= mx
    return answ

def RGB2YUV(rr,gg,bb):
    y = 0.299* rr + 0.587*gg + 0.114*bb
    u = -0.147*rr - 0.289*gg + 0.436*bb
    v = 0.615*rr - 0.515*gg - 0.100*bb
    return y,u,v
