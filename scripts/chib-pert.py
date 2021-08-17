from FICUS import AnalyticField as af
from FICUS import MagnetReader as mr
from FICUS import PlasmaReader as pr

import numpy as np
import sys

'''
Uses FICUS magnets to calculate error field on a plasma surface
This program takes the generalized 3D files

copy of bnorm-surface.py
used to read perturbed b-error

Updated 13 August 2021
'''

try:
    fin = sys.argv[1]
except:
    fin = 'zot80_3d.csv'
    print('default file:', fin)

try:
    path = sys.argv[2]
except:
    path = './'
    print('default path:', path)

f_coil = 'tf-coils-pg19-halfp.h5'
plasma = pr.ReadPlasma(f_coil)
targets = plasma.r_plasma


# may need to add surface file to compute bnormal error
try:
    mag = mr.Magnet_3D_gen(fin, HLW=True)
except:
    mag = mr.Magnet_3D_gen(fin, HLW=False)
#source = mag.export_source() # bug?
source = mag.export_source_old(HLM=True)
    #x0,y0,z0,nx,ny,nz,ux,uy,uz, H,L,M = source


t = af.Timer()
#Btot = af.calc_B(targets,source,_face=False).block_until_ready()
Btot = af.calc_B(targets,source,_face=False, B_func=af.jit_B3d).block_until_ready()
#Btot = af.calc_B(targets,source, B_func=af.jit_Bvec_dipole)


t.start('calc bnorm dipole')
chib = plasma.calc_bnorm(Btot)
t.stop()
print('chib', chib)

# saves output
tag = fin[:-4]
fout = tag + '.out'
with open(fout,'w') as f:
    print('fname,' , fin ,file=f)
    print('chib,', chib, file=f)
print('wrote to', fout)

print('bye')
sys.exit()
