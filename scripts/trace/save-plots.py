import sys
import os

# usage: python save-plots.py <list-h5-files>
# updated 19 Jan 2021

print('usage: python save-plots.py <list-h5-files>')

files = sys.argv[1:]
for f in files:

    print(' reading: %s' % f)
    #os.system('srun -n 4 -t 1:00:00 --mem=8GB python plot_fieldline.py %s &'%f)
    os.system('python fieldline-plot.py %s &'%f)
