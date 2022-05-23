import numpy as np
from FICUS import MgridIO 
import sys
import os

'''
    This program reads an netCDF
    writes the appropriate XFIELDLINES input file
    and runs the job on slurm

    Updated 23 May 2022
'''

### Read
try:
    f_mgrid = sys.argv[1]

except:
    print(' usage: python run-fieldlines.py mgrid.nc')
    sys.exit()

mgrid = MgridIO.Read_MGRID()
try:
    mgrid.read_netCDF(f_mgrid) 
except:
    print(' could not read netCDF mgrid, attempting to read binary mgrid')
    mgrid.read_binary(f_mgrid, _debug=True) 
    #mgrid.read_binary(f_mgrid) 
# make this more flexible, read the binary in try/catch?

### Edit
f_sample = 'input.sample-fieldline'
#f_sample = 'input.grid.nc-fl'
with open(f_sample) as f:
    sample_input = f.readlines()


# TODO:  need to update the NFP automatically

# find line where FIELDLINES input starts
start = 0
for line in sample_input:
    if (line.find('&FIELDLINES_INPUT') > -1):
        break
    start += 1
total = len(sample_input)

r_param = np.linspace( mgrid.rmin, mgrid.rmax, 3)
r_start = '{:.2f} {:.2f} {:.2f}'.format(*r_param)
knobs = ['NR', 'NPHI', 'NZ', 'RMIN', 'RMAX', 'ZMIN', 'ZMAX', 'R_START']
value = [mgrid.nr, mgrid.nphi, mgrid.nz, mgrid.rmin, mgrid.rmax, mgrid.zmin, mgrid.zmax, r_start]
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
                if (line.find(knob) > -1):
                    line = ' {:15} =   {},\n'.format(knob, value[j])
                    #print('  %s'%line)
            f.write(line)

        print('  writing {:40}: {}'.format(f_new, value))


tag = f_mgrid[:-3] # this is hardcoded to remove .nc
fout = 'input.' + tag
write_input(fout)


### Run
f_log = 'log.' + tag
cmd = 'srun -n 96 -t 1:00:00 xfieldlines -vmec {} -mgrid {} -vac -auto > {} &'.format(tag, f_mgrid, f_log)
print('  Runing: ', cmd)
os.system(cmd)
print(' saving log to:', f_log)

