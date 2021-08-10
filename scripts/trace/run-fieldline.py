import numpy as np
from FICUS import MgridIO 
import sys
import os

'''
    This program reads an netCDF
    writes the appropriate XFIELDLINES input file
    and runs the job on slurm

    Updated 5 July 2021
'''

### Read
try:
    f_mgrid = sys.argv[1]

except:
    print(' usage: python run-fieldlines.py mgrid.nc')
    sys.exit()

mgrid = MgridIO.Read_MGRID()
mgrid.read_netCDF(f_mgrid)


### Edit
f_sample = 'input.101-test'
with open(f_sample) as f:
    sample_input = f.readlines()

# find line where FIELDLINES input starts
start = 0
for line in sample_input:
    if (line.find( '&FIELDLINES_INPUT' ) > -1):
        break
    start += 1
total = len(sample_input)

knobs = ['NR', 'NPHI', 'NZ', 'RMIN', 'RMAX', 'ZMIN', 'ZMAX']
value = [mgrid.nr, mgrid.nphi, mgrid.nz, mgrid.rmin, mgrid.rmax, mgrid.zmin, mgrid.zmax]
N_edits = len(knobs)

def write_input(f_new):


    with open(f_new, 'w') as f:

        for i in np.arange(total):

            line = sample_input[i]
            if (i < start):
                f.write(line) 
                continue

            for j in np.arange(N_edits):
                knob = knobs[j]
                if (line.find( knob ) > -1):
                    line = ' {:15} =   {},\n'.format(knob, value[j])
                    #print('  %s'%line)
            f.write(line)

        print('  writing {:40}: {}'.format(f_new, value ) )

tag = f_mgrid[:-3]
fout = 'input.' + tag
write_input(fout)


### Run
f_log = 'log.' + tag
os.system('srun -t 1:00:00 xfieldlines -vmec {} -mgrid {} -vac -auto > {} &'.format(tag,f_mgrid,f_log) )
print(' saving log to:', f_log)

