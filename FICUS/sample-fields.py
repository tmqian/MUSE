from FICUS import AnalyticForce as af
from FICUS import MagnetReader as mr
#from FICUS import PlasmaReader as pr

import numpy as np
import sys
'''
This codes samples the field from a set of PM 
By default it generates a grid of 2N*N points on the faces of each PM.
Writes output to an 2Nx6 csv file (x,y,z, bx,by,bz)

Updated 31 May 2021
'''

path = './'
fin = 'block_zot80_3sq_1.csv'

#f_coil = 'tf-coils-pg19-halfp.h5'
#plasma = pr.ReadPlasma(f_coil)
#targets = plasma.r_plasma


# may need to add surface file to compute bnormal error
mag = mr.Magnet_3D(path+fin)

#source = mag.export_source_dipole()
source = mag.export_source()
N_source = len(source)

try:
    N_grid = int(sys.argv[1])
except:
    print('default N_grid = 1')
    N_grid = 1
targets = mag.export_target_n2(N_grid,dz=1e-6)

t = af.Timer()
Btot = af.calc_B(targets,source)#.block_until_ready()
#Btot = af.calc_B(targets,source, B_func=af.jit_Bvec_dipole)

#mask = af.mask_self_interactions(8,N_source)
#pdb.set_trace()
#Bnew = Btot.T * mask
#B2 = np.sum(Bnew.T,axis=0)
#Bx,By,Bz = B2.T

Bx,By,Bz = Btot.T
X,Y,Z = targets.T
#Bx,By,Bz,Bmag = af.split(Btot)


fout = 'B-target-v6-n%i.csv' % (2 * N_grid**2)
data = np.array([X,Y,Z,Bx,By,Bz]).T
af.write_data(data,fout)

print('bye')
sys.exit()
