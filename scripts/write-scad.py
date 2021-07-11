#!/usr/bin/env python
# coding: utf-8

'''
    usage: python script.py fname.focus N_slieces[default=18]
    last updated : 11 July 2021

    This file extends wyrm-finder.py
    It finds
        spaces (where cavity height exceeds tower height)
        gaps   (where a single tower has space in between magnets)
        gaps with opposite sign (previously called wyrms)

    There are 3 user flags:
        _show opens up plot in X displace
        _save records plot as .png
        _write records the indices of each category (which may be used in other codes)
'''

_show  = True
_save  = False
_write = False

# In[1]:

import numpy as np
import sys
import matplotlib.pyplot as plt

try: 
    import MagnetReader as mr
except:
    from scripts import MagnetReader as mr


### main
print('Welcome to FAMUS Wyrm Finder')

# open file
try:
    fname = sys.argv[1]
    fd = mr.ReadFAMUS(fname)
    Pho = fd.pho
#    with open(fname) as f:
#        data_in = f.readlines()
    
except:
    print('  error: issue finding file')
    print('  usage: python script.py fname.focus N_slices[default=18]')
    sys.exit()
    
print('Read file %s' % fname)


# load data

#data0 = np.array([line.strip().split(',') for line in data_in[3:]])
#try: # some files end with an extra comma
#    data1 = np.array((data0[:,3:]),float)
#    X,Y,Z,Ic,M,Pho,Lc,Mp,Mt = np.transpose(data1)
#except:
#    data1 = np.array((data0[:,3:-1]),float)
#    X,Y,Z,Ic,M,Pho,Lc,Mp,Mt = np.transpose(data1)

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


def norm(v):
    v = np.array(v)
    return v / mag(v)

def mag(v):
    return np.sqrt( np.sum(v*v) )

# given list of xyz vectors, return nhat relative to a torus
def xyz_to_n(xyz,R0=0.3048):
    
    nhat = []
        
    N = len(xyz)
    for k in np.arange(N):
        
        x,y,z = xyz[k]
        u = np.arctan2(y,x)
        
        x0 = R0 * np.cos(u)
        y0 = R0 * np.sin(u)
        z0 = 0

        r  = xyz[k]
        r0 = np.array( [x0,y0,z0] )
        n = norm(r-r0)
        
        nhat.append(n)
        
    return np.array(nhat)

# tower-height
def write_scad(data, fname='test.txt'):
    # format: data = np.transpose((X,Y,Z,u,v,w,H))
    
    with open(fname,'w') as f:
        f.write('N_DIP = %i; \n'% len(data) )
        f.write('data = [')
        for line in data[:-1]:
            if line[-1]==0:
                continue
            f.write('[ {}, {}, {}, {:.6}, {:.6}, {:.6}, {}],\n'.format(*line))

        f.write('[ {}, {}, {}, {:.6}, {:.6}, {:.6}, {}]];'.format(*(data[-1]) ))
        
  

# manipulate data
X, Y, Z, Ic, M, Pho, Lc, MP, MT = np.transpose(fd.data[:N_towers])

# go to UV space
U = np.arctan2(Y,X)
R = np.sqrt(X*X + Y*Y)
R0 = 0.3048
A = np.sqrt( (R-R0)**2 + Z**2 )
V = np.arctan2(Z, R-R0)

xyz = np.transpose([X,Y,Z])
nhat = xyz_to_n(xyz)

# copy (maybe this still writes slices, instead of towers)
u,v,w = nhat.T

# get toroidal angle
U = np.arctan2(Y,X)

# split into half pipes
idx1 = np.array( U < np.pi/4 , int ) * np.array( U > -np.pi/4 , int )
idx2 = np.array( U > np.pi/4 , int ) * np.array( U < 3*np.pi/4 , int )



# unfold data into tower map
Pho = fd.pho
mat = np.reshape(Pho,(N_slices,N_towers)).T
arg = np.reshape(np.arange(n_dip), (N_slices,N_towers)).T

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


# new code

def find_base(tower):

    if np.max(tower) == 0:
        return 0

    count = 0
    offset = 4 # hard-coded
    for s in tower[offset:]:
        if (s > 0):
            break
        count += 1

    return count

mask = np.arange(N_slices) + 1 - 4
a = np.abs(mat) * mask[np.newaxis,:]
height = np.max(a, axis=1)
mass = np.sum( np.abs(mat), axis=1 )
base = np.array([ find_base(t) for t in a ])
arg_space = np.argwhere( height - mass > 0)
arg_gap = np.argwhere( height - mass - base > 0)

print('n towers w gaps       :', len(arg_gap) )
print('n towers w spaces     :', len(arg_space) )

# check
#(np.sum(idx1) + np.sum(idx2))/2
data = np.transpose((X,Y,Z,u,v,w,height))

mag1 = data[ np.argwhere(idx1) ][:,0]
mag2 = data[ np.argwhere(idx2) ][:,0]

#stop = 9999 # use this if more than 9999 towers, because openSCAD array size maxes out at 9999
#write_scad(mag1[:stop], fname='0711-scad-x45.txt')
#write_scad(mag1[stop:], fname='0711-scad-y45.txt')
write_scad(mag1, fname='0711-scad-x45.txt')
write_scad(mag2, fname='0711-scad-y45.txt')

import pdb
pdb.set_trace()

sys.exit()

# prepare plots

def plot_towers(args, tag='both'):
    pos_points = (above[ args ] * np.arange(N_slices))[:,0].T
    neg_points = (below[ args ] * np.arange(N_slices))[:,0].T
    arg_points = arg[ args ][:,0]
    
    
    plt.figure(figsize=(12,6))
    for j in np.arange(N_slices):
        plt.plot(pos_points[j]+1,'r.')
        plt.plot(neg_points[j]+1,'b.')
    plt.title( '%i towers w %s :: %s ' % ( len(args), tag, fname ) )
    plt.yticks( np.arange(1,20) )
    plt.ylim(3,20)
    plt.ylabel('layer number')
    plt.xlabel('tower index')
    plt.grid()
    plt.draw()
#    plt.show()
    if (_write):
        csv  = 'w_%s-map-%s.csv' % (fname[:-6], tag)
        np.savetxt(csv, arg_points, delimiter=',',fmt='%i')
        print('Wrote to file %s' % csv)
    if (_save):
        fout = 'w_%s-map-%s.png' % (fname[:-6], tag)
        plt.savefig(fout)
        print('Wrote to file %s' % fout)

arg_both = np.argwhere(both > 0)
plot_towers(arg_both)
plot_towers(arg_space, tag='spaces')
plot_towers(arg_gap, tag='gaps')

if (_show):
    plt.show()

import pdb
pdb.set_trace()


sys.exit()
