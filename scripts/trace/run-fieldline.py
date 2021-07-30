import numpy as np
import sys
import os

# usage: python run-fieldlines.py <list-mgrid-files>
# updated 19 Jan 2021

files = sys.argv[1:]
for f in files:

        mgrid = f
        vmec = f[12:] + '-fl'
        print(' launching: %s' % mgrid)

        # run on cluster
        os.system('srun -N 2 -t 2:00:00 --mem=8GB xfieldlines -vmec %s -mgrid %s -vac -auto &'%(vmec, mgrid))  # for eddy
        #os.system('srun -n 4 -t 2:00:00 --mem=8GB xfieldlines -vmec %s -mgrid %s -vac -auto &'%(vmec, mgrid)) # for portal

        # run local
        #cmd = 'xfieldlines -vmec {} -mgrid {} -vac -auto >| log.{} &'.format(vmec, mgrid, vmec)
        #print(cmd)
        #os.system(cmd)

