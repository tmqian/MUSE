#!/usr/bin/env python
# coding: utf-8

'''
    usage: python script.py fname.focus N_slices')
    last updated : 5 Feb 2021
'''

_save = True

# In[1]:

import numpy as np
import sys
import matplotlib.pyplot as plt
'''
    + compresses magnetization to the bottom t* style remainders
    + leaves everything non-zero to enable further optimization
    + preserves net sign
    + annihilates bubbles
    + but preserves gaps at the base
'''
def compress(tower):
    
    #global N_slices
    new = np.ones(N_slices)*1e-4

    mag = abs(np.sum(tower))
    if (mag > 0):
        sgn = np.sum(tower)/mag
    else:
        sgn = 1

    k = N_base-1 # need to include base
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



def compute_base(Ic):

    mat = np.reshape(Pho,(N_slices,N_towers))
    isEmpty = np.array( np.sum(mat,axis=1) == 0 )
    base = np.sum(isEmpty)
    return base


### main
print('Welcome to FAMUS Wyrm Finder')

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


# load data

data0 = np.array([line.strip().split(',') for line in data_in[3:]])
try: # some files end with an extra comma
    data1 = np.array((data0[:,3:]),float)
    X,Y,Z,Ic,M,Pho,Lc,Mp,Mt = np.transpose(data1)
except:
    data1 = np.array((data0[:,3:-1]),float)
    X,Y,Z,Ic,M,Pho,Lc,Mp,Mt = np.transpose(data1)


# set tower height, calculate number of towers
try:
    N_slices = int( sys.argv[2] )
    print('  user selected tower height: {}'.format(N_slices) )
    
except:
    N_slices = 18
    print('  default tower height: %i slices' % N_slices)

n_dip = len(Pho)
N_towers = int(n_dip / N_slices)
print('  N Dipoles, Slices, Towers: {} {} {}'.format(n_dip,N_slices,N_towers) )


# unfold data into tower map
mat = np.reshape(Pho,(N_slices,N_towers)).T
arg = np.reshape(np.arange(n_dip), (N_slices,N_towers)).T
N_base = compute_base(Ic)

above = np.array( mat > 0, int )
below = np.array( mat < 0, int )
print('n slices positive :', np.sum(above) )
print('n slices negative :', np.sum(below) )

pos_tow = np.sum(above, axis=1)
neg_tow = np.sum(below, axis=1)
both = np.array( pos_tow*neg_tow != 0, int)
print('n towers w pos slices :', np.sum( np.array(pos_tow>0, int) ) )
print('n towers w neg slices :', np.sum( np.array(neg_tow>0, int) ) )
print('n towers w both       :', np.sum(both) )

# prepare plots
pos_points = (above[ np.argwhere(both>0)] * np.arange(N_slices))[:,0].T
neg_points = (below[ np.argwhere(both>0)] * np.arange(N_slices))[:,0].T
arg_points = arg[ np.argwhere(both>0)][:,0]
if (_save):
    csv = 'wyrm-map-%s.csv'% sys.argv[1][:-6]
    np.savetxt(csv, arg_points, delimiter=',',fmt='%i')
    print('Wrote to file %s' % csv)
plt.figure(figsize=(12,6))
for j in np.arange(N_slices):
    plt.plot(pos_points[j]+1,'r.')
    plt.plot(neg_points[j]+1,'b.')
plt.title( '%i towers w both :: %s ' % (np.sum(both), sys.argv[1]) )
plt.yticks( np.arange(1,20) )
#xticks = np.arange(1,len(both))
#plt.axes().yaxis.set_minor_locator(xticks)
plt.ylim(3,20)
plt.ylabel('layer number')
plt.xlabel('tower index')
plt.grid()
fout = 'wyrm-map_'+ file.split('/')[-1][:-6] + '.png'

if (_save):
    plt.savefig(fout)
    print('Wrote to file %s' % fout)
plt.draw()
plt.show()
