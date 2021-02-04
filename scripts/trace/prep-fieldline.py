import numpy as np
import sys
import os

# usage: python run_fieldlines.py <list-mgrid-files>
# updated 19 Jan 2021

print('usage: python run_fieldlines.py <list-mgrid-files>')

files = sys.argv[1:]
for f in files:

    fname = 'input.' + f[12:] + '-fl'
    print(' writing: %s' % fname)
    os.system('cp scripts/trace/input.sample-fieldline %s' % fname)
