#!/usr/bin/env python
# coding: utf-8

'''
This script reads a FAMUS fname.focus file
modifies the dipoles to take values pho = (-1,0,1)
and writes to FAMUS file bin_fname.focus, keeping all else the same.

usage: 
$ python script.py fname.focus

on Eddy it may be necessary to 
$ module load anaconda3

Last Updated: 8 Oct 2020 by Tony Qian
'''

# In[1]:

import numpy as np
import sys


# In[2]:

print('Welcome to FAMUS Binary Dipoles')

# open file
try:
    file = sys.argv[1]
    with open(file) as f:
        data_in = f.readlines()
    
except:
    print('  error: issue finding file')
    print('  usage: python script.py fname.focus cut')
    sys.exit()
    
print('Read file %s' % file)


# set rounding cutoff
try:
   cut = float(sys.argv[2])
   print('  using round threshold: %.2f' % cut)
except:
   print('  using default cut: 0.9')
   cut = 0.9


# In[3]:

# load data
data0 = np.array([line.strip().split(',') for line in data_in[3:]])
data1 = np.array((data0[:,3:-1]),float)

X,Y,Z,Ic,M,Pho,Lc,Mp,Mt = np.transpose(data1)


# In[4]:

# force binary dipoles
bin_pho = np.array( Pho > cut, int ) - np.array( Pho < -cut, int )
data_out = np.transpose([ X,Y,Z,Ic,M,bin_pho,Lc,Mp,Mt])


# In[5]:


#write

n_dip = len(Pho)
dname = [ 'pm{:08d}'.format(j) for j in np.arange(n_dip) ]

#path = file.split('/')[:-1]+'/'
fout = 'rnd%i_'%(10*cut) + file.split('/')[-1]

with open(fout,'w') as f:
    
    for j in np.arange(3):
        f.write(data_in[j])

    for j in np.arange(n_dip):
        x,y,z, ic,m,bpho,lc,mp,mt = data_out[j]
        line = '{:4}, {:4}, {:13}, {:15.8e}, {:15.8e}, {:15.8e}, {:2d}, {:15.8e}, {:15.8e}, {:2d}, {:15.8e}, {:15.8e},\n'.format(
            2,2,dname[j],
            x,y,z, 
            int(ic),m,bpho,
            int(lc),mp,mt )
        f.write(line)


print('Wrote to file %s' % fout)

