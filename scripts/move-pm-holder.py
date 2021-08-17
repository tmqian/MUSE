from scripts import MagnetReader as mr
import numpy as np
import sys

'''
   This program adds correlated perturbations to 3D magnet files
   Pipe 1 and 3 are - X holders
   Pipe 2 and 4 are - Y holders

   Updated by Tony Qian and Dominic Seidita 
   13 August 2021
'''


#f_focus = sys.argv[1]
f_focus = 'zot80_3d.csv'
#mag = mr.ReadFAMUS(f_focus)
#mag = mr.Magnet_3D(f_focus)
mag = mr.Magnet_3D_gen(f_focus) # reads 3D csv



x,y,z = mag.r0.T
#x,y,z = mag.com.T
phi = np.arctan2(y,x)

pi4 = np.pi/4
pipe1 = np.argwhere( np.abs(phi) < pi4)         [:,0]
pipe2 = np.argwhere( np.abs(phi - 2*pi4) < pi4) [:,0]
pipe4 = np.argwhere( np.abs(phi + 2*pi4) < pi4) [:,0]
pipe3 = np.argwhere( np.abs(phi) > 3*pi4)       [:,0]

### make a change

dr = 0.005
pert1 = dr*np.array([1,0,0])
pert2 = dr*np.array([0,1,0])
pert3 = dr*np.array([0,0,-1])

mag.r0[pipe1] += pert1
mag.r0[pipe2] += pert2
mag.r0[pipe3] += pert3


# write
try:
    fout = sys.argv[1]
except:
    print(' using default name')
    fout = 'mod-zot80_3d.csv'
mag.write_magnets(fout)
