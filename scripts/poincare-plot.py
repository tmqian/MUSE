#!/usr/bin/env python
# coding: utf-8

'''
Updated: 21 Jan 2021
This script reads FAMUS/FOCUS output and saves a poincare plot

usage: python3 script.py focus-fname.h5 key

where "key" is an integer (0,1,2,3) which saves .pdf and .png in binary convention

0 = 00 : neither
1 = 01 : pdf
2 = 10 : png
3 = 11 : both

takes an optional plasma file for plotting boundary
'''


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys

from netCDF4 import Dataset
#from coilpy import FourSurf
import MagnetReader as mr

# In[2]:


def get(f,key):
    return f.variables[key][:]


# In[3]:

print('Welcome to Poincare Plotter')

try:
    file = sys.argv[1]
    f = Dataset(file, mode='r')
except:
    print('  error: issue finding file')
    print('  usage: python script.py FOCUS-fname.h5')
    sys.exit()
fname = file.split('/')[-1][6:-3]
print('Read file %s' % fname)

# In[5]:

try:
    ns = get(f,'pp_ns')
    r = get(f,'ppr')
    z = get(f,'ppz')
    r0,z0 = get(f,'pp_raxis'),get(f,'pp_zaxis')
except:
    print('  error: issue reading file')
    print('  perhaps .h5 did not finish writing poincare plot')
    sys.exit()

ns = int(ns)

# In[7]:


Bn = get(f,'Bn')
nn = get(f,'nn')
nfp = get(f,'Nfp')
n_theta = get(f,'Nteta')
n_zeta = get(f,'Nzeta')


# In[8]:

# calculate chi2B = \int (B \cdot n)**2 dA
bnormal = np.sum((1/2) * np.array(Bn*Bn*nn) * (2*np.pi / n_theta)*(2*np.pi/(2*nfp) / n_theta))


# In[9]:


plt.figure()

colors = cm.rainbow(np.linspace(1, 0, ns))

for i in np.arange(ns):
    plt.plot(r[:,i], z[:,i],'.', color=colors[i],markersize=1)

plt.plot(r0,z0,'.',color='gray',label=r'$\chi^2_B$ = %.2e' % bnormal)

plt.axis('square')

plt.ylim(-0.09,.09)
plt.xlim(.2,.38)

plt.ylabel('Z [cm]')
plt.xlabel('R [cm]')

plt.title(fname)
plt.legend(loc=2)

# plasma plot

def plasma_plot(zeta=0,npoints=360):
    rb,zb = plasma.rz(np.linspace(0, 2*np.pi, npoints), zeta * np.ones(npoints))
    plt.plot(rb,zb,'tab:gray',ls='--',lw=.7)
 
try:
    fboundary = sys.argv[3]
    print(' plasma file found: ', fboundary)
    plasma = mr.FourSurf.read_focus_input(fboundary)
    plasma_plot()

except:
    #print('plasma issue')
    pass

# write
    
def write(fout):
    plt.savefig(fout)
    #plt.savefig(fout,bbox=False)
    print('Wrote Poincare Plot to file: %s' % fout)

try:
    key = int(sys.argv[2])
except:   
    print(' No write command given')
    print('  usage: python script.py focus-fname.h5 key')
    print('    key=1 : writes .pdf')
    print('    key=2 : writes .png')
    print('    key=3 : writes .pdf and .png')
    #sys.exit()
    print(' using default: key=2, .png')
    key = 2
    
if (key == 1):
    fout = 'pp-%s.pdf' % fname
    write(fout)
    
elif (key == 2):
    fout = 'pp-%s.png' % fname
    write(fout)
    
elif (key == 3):
    fout = 'pp-%s.pdf' % fname
    write(fout)
    fout = 'pp-%s.png' % fname
    write(fout)
    
else:
    print('  recieved incorrect write command: %s' % key)
    print('  usage: python script.py focus-fname.h5 key')
    print('    key=1 : writes .pdf')
    print('    key=2 : writes .png')
    print('    key=3 : writes .pdf and .png')
 


