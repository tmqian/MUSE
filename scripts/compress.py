#!/usr/bin/env python
# coding: utf-8

'''
    usage: python script.py fname.focus N_slices')
    last updated : 3 Jan 2021
'''

# In[1]:

import numpy as np
import sys

print('Welcome to FAMUS Dipole Compressor')

# In[2]:

# open file
try:
    file = sys.argv[1]
    with open(file) as f:
        data_in = f.readlines()
    
except:
    print('  error: issue finding file')
    print('  usage: python script.py fname.focus N_slices')
    sys.exit()
    
print('Read file %s' % file)



# In[3]:

# load data

data0 = np.array([line.strip().split(',') for line in data_in[3:]])
try: # some files end with an extra comma
    data1 = np.array((data0[:,3:]),float)
    X,Y,Z,Ic,M,Pho,Lc,Mp,Mt = np.transpose(data1)
except:
    data1 = np.array((data0[:,3:-1]),float)
    X,Y,Z,Ic,M,Pho,Lc,Mp,Mt = np.transpose(data1)



# In[4]:

# set tower height, calculate number of towers
try:
    N_slices = int( sys.argv[2] )
    print('  user selected tower height: {}'.format(N_slices) )
    
except:
    N_slices = 16
    print('  default tower height: %i slices' % N_slices)

n_dip = len(Pho)
N_towers = int(n_dip / N_slices)
print('  N Dipoles, Slices, Towers: {} {} {}'.format(n_dip,N_slices,N_towers) )

# unfold data into tower map
mat = np.reshape(Pho,(N_slices,N_towers)).T


# In[5]:

'''
    + compresses magnetization to the bottom t* style remainders
    + leaves everything non-zero to enable further optimization
    + preserves net sign
    + annihilates bubbles
    + but preserves gaps at the base
'''
def compress(tower):
    
    global N_slices
    new = np.ones(N_slices)*1e-4

    mag = abs(np.sum(tower))
    if (mag > 0):
        sgn = np.sum(tower)/mag
    else:
        sgn = 1

    k = 0    
    if (mag < 1): 
        # empty tower
        new[k] = mag 
        
    else:        
        # filled tower, check for bubbles at base


        while ( abs(tower[k]) < 0.1):
            k += 1
            # rest if no bubbles found
            if (k == N_slices):
                k=0
                break
      
        # start filling and keep remainder
        while (mag > 1):
            
            new[k] = 1
            k     += 1
            mag   -= 1
        
        if (k < N_slices):    
            new[k] = mag
        
    return new * sgn


# compress towers

com = np.array([ compress(mat[j]) for j in np.arange(N_towers)])

com_pho = np.ravel(com.T) * Ic    # ignores magnets which are switched off

data_out = np.transpose([ X,Y,Z,Ic,M,com_pho,Lc,Mp,Mt])

dname = [ 'pm{:08d}'.format(j) for j in np.arange(n_dip) ]


# In[6]:

# write

fout = 'com_'+ file.split('/')[-1]

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

