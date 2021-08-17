from FICUS import AnalyticField as af     # Calculate field from 3D Dipoles 
from FICUS import MagnetReader as mr      # Read FAMUS dipoles
from FICUS import CoilField as cf         # Calculate Coil Field
from FICUS import MgridIO as mg           # Handles MGRID file I/O
#from FICUS import PlasmaReader as pr

import numpy as np
import sys
'''
This codes reads sources from a set of PM + TF coils
and evaluates the field on targets set up by MGRID

Then it writes the field to an MGRID netCDF

Changed output file name to match the input file. Code now asks for input file for trim coils.
Made compatible with magnet-to-poincare.py

Updated 12 August 2021
'''

path = './'
fin = 'block_zot80_3sq_1.csv'        # 3D Dipole (engineering)
f_pmag = 'zot80_3d.csv'              # 3D Dipole (physics)
f_coil = 'phased-tf-433A-33N.ficus'  # FICUS Coils (BUSTED)
f_trim = sys.argv[1]

#f_coil = 'tf-coils-pg19-halfp.h5'
#plasma = pr.ReadPlasma(f_coil)
#targets = plasma.r_plasma


### Sets MGRID parameters (cylindrical coordinates)

NR = 101      # 101 number radial points
NZ = 101      # 101 number z points
NPHI = 72     # 72 number of azimuthal points, per field period
RMIN = 0.22   
RMAX = 0.38
ZMIN = -0.08
ZMAX = 0.08

tag = f_trim[:-4]
fout = tag + '.nc'  # MGRID output filename

# PM source
mag    = mr.Magnet_3D(path+fin)
source = mag.export_source()
#mag = mr.Magnet_3D_gen(path+f_pmag)
#source = mag.export_source_old()


# TF source (write this as an import function in CF)
with open(f_coil) as f:
    datain = f.readlines()
coils = np.array([ line.strip().split(',') for line in datain], float)


# Trim coil source
with open(f_trim) as f:
    datain = f.readlines()
trim_coils = np.array([ line.strip().split(',') for line in datain], float)


# Targets
mgrid   = mg.Read_MGRID(nr=NR,nz=NZ,nphi=NPHI,rmin=RMIN,rmax=RMAX,zmin=ZMIN,zmax=ZMAX)
targets = mgrid.init_targets()


# Evaluate B
B_trim = af.calc_B(targets,trim_coils,_face=False, B_func=af.jit_B3d, n_step=1000)
B_coil = cf.calc_B(targets,coils)
#B_trim = af.jit_B3d(targets,trim_coils)
B_pmag = af.calc_B(targets,source,_face=False, B_func=af.jit_B3d, n_step=1000)


# debugging: saves output as csv, in addition to mgrid
def save(fname,data):
    with open(fname,'w') as f:
        for line in data:
            print('{}, {}, {}'.format(*line),file=f )

    print(' Wrote ', data.shape, ' to ', fname)

mgrid.add_field(B_pmag, 'PM field')
mgrid.add_field(B_coil, 'TF field')
mgrid.add_field(B_trim, 'Trim Coil field')

mgrid.write(fout)

print('bye')
sys.exit()


