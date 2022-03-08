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
#f_sample = 'input.101-test'
path = '/home/tqian/CODE/MUSE/scripts/trace/'
f_sample = path + 'input.sample-fieldline'
with open(f_sample) as f:
    sample_input = f.readlines()

# find line where FIELDLINES input starts
start = 0
for line in sample_input:
    if (line.find( '&FIELDLINES_INPUT' ) > -1):
        break
    start += 1
total = len(sample_input)

#knobs = ['NR', 'NPHI', 'NZ', 'RMIN', 'RMAX', 'ZMIN', 'ZMAX', 'R_START']
#rstart = ' {}  {}  {} '.format( mgrid.rmin, (mgrid.rmin + mgrid.rmax)/2, mgrid.rmax )
#value = [mgrid.nr, mgrid.nphi, mgrid.nz, mgrid.rmin, mgrid.rmax, mgrid.zmin, mgrid.zmax, rstart]
knobs = ['NR', 'NPHI', 'NZ', 'RMIN', 'RMAX', 'ZMIN', 'ZMAX']
value = [mgrid.nr, mgrid.nphi, mgrid.nz, mgrid.rmin, mgrid.rmax, mgrid.zmin, mgrid.zmax]
N_edits = len(knobs)

def write_input(f_new):

    with open(f_new, 'w') as f:

        for i in np.arange(total):

            line = sample_input[i]
            if (i < start):

                # &VMEC (indata) edits
                if (line.find( 'NFP' ) > -1):
                    line = '  {:15} =   {},\n'.format('NFP', mgrid.nfp )

                if (line.find( 'EXTCUR' ) > -1):
                    # condition on 'S' vs 'R'?

                    if (mgrid.mode == 'S'):
                        extcur = "".join( [ " {} ".format(I) for I in mgrid.raw_coil_current])
                    else:
                        extcur = "".join( [ " {} ".format(1) for I in mgrid.raw_coil_current])
                    line = '  {:15} =   {},\n'.format('EXTCUR', extcur )

                f.write(line) 
                continue


            ## &fieldline edits
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
#cmd = 'srun -N 1 -t 2:00:00 xfieldlines -vmec {} -mgrid {} -vac  > {} &'.format(tag,f_mgrid,f_log)  # it seems that auto fills in field line, no auto just does 3 points
cmd = 'srun -N 1 -t 2:00:00 xfieldlines -vmec {} -mgrid {} -vac -auto > {} &'.format(tag,f_mgrid,f_log) 
os.system(cmd)
print('   running:', cmd)
print('   saving log to:', f_log)

