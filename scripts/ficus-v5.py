#!/usr/bin/env python
# coding: utf-8

# Updated 7 March 2021

'''
    usage: python famus-one-iter.py
'''

from coilpy import *

import numpy as np
import matplotlib.pyplot as plt

import sys
import MagnetReader as mr

_debug = False
_write = False

### USER INPUT ###

N_layers = 18

##################

def load_rings(self, N_poloidal=89):
    
    u,v = self.to_uv()
    v_idx = np.array(v*100,int)

    rings = []
    j = 0
    while j < N_towers:
        
        count = 1
        while ( np.abs(v[j+1] - v[j]) < 1./N_poloidal):
            count += 1
            j += 1
            
        rings.append(count)
        j += 1
        
    return rings

'''
    u: j toroidal
    v: i poloidal
    w: k slice

    note - (u,v) starts from 0 while w starts from 1
    such that w=5 points to the 5th layer
'''
def idx_3d(u,v,w):
    return u + np.sum(rings[:v]) + N_towers*(w-1)

def idx_2d(u,v):
    return u + np.sum(rings[:v])

def show_tower(self,u,v):
    
    idx = idx_2d(u,v)
    tow = np.reshape(self.pho,(N_layers,N_towers)).T[idx]
    print('({},{}):'.format(v,u), tow)


# reads instruction set, allowing for blank lines and comments ('#')
def read_txt(fin):

    with open(fin) as f:
        datain = f.readlines()

    data = []        
    for line in datain:
        if (line.find('#') > -1): # comment
            continue
        if (not line.strip()):    # blank
            continue
        data.append( line.strip().split(',') )

    return np.array(data, int)


### Load plasma surface

f_coil  = 'tf-coils-pg19-halfp.h5'
h5_coilsurf = FOCUSHDF5(f_coil)
    


### load dipoles

try:
    f_famus = sys.argv[1]
except:
    print('  usage: python ficus.py fname.focus')
    sys.exit()

#open new file
print('Loading famus file: ', f_famus)
fd = mr.ReadFAMUS(f_famus)
print('  read N dipoles:', len(fd.X) )

#! add a feature that protects potential zots from skim

# set up coordinates
N_dipoles = len(fd.X)
N_towers  = int(N_dipoles / N_layers)


### load zot inputs

try:
    fin = sys.argv[2]
    data = read_txt(fin)
    _scan = True

except:
    print('  second file not given, or error reading file: skipping scan')
#    print('  usage: python tower-print.py fname.focus list.txt')
    _scan = False
    #sys.exit()

# I have a list (u,v,w). I want to know which idx it points to in famus\
# and I want to know which idx it points to in the skim

def make_key(data,pd):
    # get idx of target magnets
    famus_idx = []
    for line in data:
        v,u,w,m = line

        idx = pd.idx_3d(u,v,w)
        famus_idx.append(idx)

    #print('  found %i changes' % len(data) )

    # emulate skim
    i = 0
    key = []
    for j in np.arange(pd.N_dipoles):
        
        if (j in famus_idx):
            k = np.argwhere( famus_idx-j == 0 )[0,0]
            key.append([k,i,j])
            #print('k,i,j',k,i,j) # helpful for debugging

        if (pd.pho[j] != 0 or j in famus_idx):
            # preserved in skim
            i+=1

    key = np.array(key)
    arg = np.argsort(key,axis=0).T[0]
    return key[arg]


if (_scan):
    key = make_key(data,fd) 
    fd.skim(save=key[:,2])
else:
    fd.skim()

print('  after skimming N dipoles:', len(fd.X) )

#ox, oy, oz, Ic, mm, pho, Lc, mp, mt = np.transpose(fd.data)
ox = fd.X
oy = fd.Y
oz = fd.Z
mm = fd.M
pho = fd.pho
mp = fd.MP
mt = fd.MT
q = fd.q

# build dipole
mx = mm*np.sin(mt)*np.cos(mp)
my = mm*np.sin(mt)*np.sin(mp)
mz = mm*np.cos(mt)

rho = np.abs(mm)*pho**q # unsused

# prepare for inductance matrix
r_dipole = np.transpose([ox,oy,oz])
m_dipole = np.transpose([mx,my,mz])


### load plasma surface

# use theta zeta grid to extract Cartesian positions on plasma surface
px = np.ravel(h5_coilsurf.xsurf)
py = np.ravel(h5_coilsurf.ysurf)
pz = np.ravel(h5_coilsurf.zsurf)

# extract components of normal field vector computed by FOCUS
nx = np.ravel(h5_coilsurf.nx)
ny = np.ravel(h5_coilsurf.ny)
nz = np.ravel(h5_coilsurf.nz)

# Load coil field
Bn_coil  = np.ravel(h5_coilsurf.Bn)
jacobian = np.ravel(h5_coilsurf.nn)

# apply symmetry to coil field
x = Bn_coil
coil_symm = np.concatenate((x,-x,x,-x))

# apply stellarator symmetry to plasma surface
px,py,pz,coil_symm = mr.stellarator_symmetry(px,py,pz,Bn_coil)
nx,ny,nz,jacob_symm = mr.stellarator_symmetry(nx,ny,nz,jacobian)

r_plasma = np.transpose([px,py,pz])
n_plasma = np.transpose([nx,ny,nz])

N_theta = h5_coilsurf.Nteta
N_zeta  = h5_coilsurf.Nzeta

N_plasma = len(r_plasma)
N_dipole = len(r_dipole)

print('Loading Complete')
print('  N_dipoles:', N_dipole)
print('  N_surface:', N_plasma)


### Compute Inductance
print('Computing Inductance Matrix')

#print('  calculate coupling: |r_plasma - r_dipole|')
print('  calculate coupling: r_plasma - r_dipole')
rij = r_plasma[:,np.newaxis] - r_dipole
#rij_mod = np.linalg.norm(rij,axis=2)
#r5 = rij_mod**5
#r3 = rij_mod**3


# updated function produces a matrix with orientation of m dotted in
# (N,M) instead of (N,M,3)
def g_sub(i):

    rn_i = np.sum( rij[i] * n_plasma[i], axis=1) 
    rm_i = np.sum( rij[i] * m_dipole, axis=1) 
    nm_i = np.sum(n_plasma[i] * m_dipole, axis=1) 

    rmod = np.linalg.norm(rij[i], axis=1)

    A = 3 * rn_i * rm_i / rmod**5
    B = nm_i / rmod**3
    #A = 3 * rn_i * rm_i / r5[i]
    #B = nm_i/r3[i] 
    return (A - B)/1e7

print('  calculate inductance: g_ij = 3 n.r r.m / r^5 - n.m / r^3')
gij = np.array([ g_sub(i) for i in np.arange(N_plasma)])
print('  inductance matrix complete')



### Given magnets, calculate field

# requires global gij matrix for bnorm, and global mm vector for pmvol
def calc_bnorm(pho):

    Bn_dipole = np.sum(gij * pho, axis=1) # uses new g_sub function, which levearges geometry
    
    # apply stellarator symmetry
    x = Bn_dipole
    dipole_symm = np.sum((x,-np.roll(x,N_theta*N_zeta),np.roll(x,N_theta*N_zeta*2),-np.roll(x,N_theta*N_zeta*3)), axis=0)
    
    # integrate residual error field 
    Bn_symm = coil_symm + dipole_symm
    bnorm = np.sum(Bn_symm*Bn_symm * np.abs(jacob_symm)) * np.pi**2/(N_theta*4*N_zeta)/2 

    # calculate pmvol
    pmvol = np.sum( np.abs(mm*pho) ) * 4
    return bnorm, pmvol

def calc_bnorm3(pho):

    Bn_dipole = np.sum(gij * pho, axis=1) # uses new g_sub function, which levearges geometry
    
    # apply stellarator symmetry
    x = Bn_dipole
    dipole_symm = np.sum((x,-np.roll(x,N_theta*N_zeta),np.roll(x,N_theta*N_zeta*2),-np.roll(x,N_theta*N_zeta*3)), axis=0)
    
    # integrate residual error field 
    Bn_symm = coil_symm + dipole_symm
    bnmax = np.max(np.abs(Bn_symm))
    bnorm = np.sum(Bn_symm*Bn_symm * np.abs(jacob_symm)) * np.pi**2/(N_theta*4*N_zeta)/2 

    # calculate pmvol
    pmvol = np.sum( np.abs(mm*pho) ) * 4
    return bnorm, pmvol, bnmax

b0,p0, bm0  = calc_bnorm3(pho) # bnorm and pmvol
print('Summary: fname, bnorm, pmvol, max(B.n)')
print('       : {}, {}, {}, {}'.format(f_famus, b0, p0, bm0))

if (not _scan): 
    sys.exit()

N_edits = len(data)
print('Loading %i changes:' % N_edits )
#print('  (v,u,w) m -> n :  integer change in magnets, relative change in bnorm (b/b0-1)')
print('  (v,u,w) m -> n :  dM, max(B.n), d(bnorm_increment), d(bnorm_total)' )

def scan_parallel():
    print('  Independent change scan')    
    pbase = np.sum(np.abs(pho))
    for j in np.arange(N_edits):
        i1,i2,i3 = key[j]
        v,u,w,m = data[j]
    
        temp = pho.copy()
        temp[i2] = m
    
        bnorm,pmvol,bnmax = calc_bnorm3(temp) # bnorm and pmvol

        string = '  edit {}: ({}, {}, {:2}) {:3} -> {:2}'.format(i1+1,v,u,w, int(pho[i2]), m)
        bchange = bnorm/b0 - 1
        pchange  = int(np.sum(np.abs(temp)) - pbase)
        print( string + ' : {}, {:10.3e}, {:10.3e}'.format(pchange, bnmax/bm0-1, bchange) )

def scan_series():
    
    print('  Accumulating change scan')    
    pbase = np.sum(np.abs(pho))
    bbase = b0
    temp = pho.copy()
    for j in np.arange(N_edits):
        i1,i2,i3 = key[j]
        v,u,w,m = data[j]
    
        temp[i2] = m
    
        bnorm,pmvol,bnmax = calc_bnorm3(temp) # bnorm and pmvol
        string = '  edit {}: ({}, {}, {:2}) {:3} -> {:2}'.format(i1+1,v,u,w, int(pho[i2]), m)

        bchange = (bnorm - bbase)/b0 
        bbase = bnorm
        btotal = bnorm/b0 - 1
        pchange  = int(np.sum(np.abs(temp)) - pbase)
        print( string + ' : {}, {:10.3e}, {:10.3e}, {:10.3e}'.format(pchange, bnmax/bm0-1, bchange, btotal) )
   
scan_parallel()
scan_series()

sys.exit()

####
N_edits = len(data)
out = []


temp = m_dipole.copy()
for j in np.arange(N_edits):
    v,u,w,m = data[j]
    idx = id3[j]

    mtemp = temp[idx].copy()
    temp[idx] = m * mtemp

    bnorm,pmvol = calc_bnorm(temp)
    #print(j,idx,u,v,w,bnorm,pmvol)
    out.append([j,idx,v,u,w,bnorm,pmvol])
    #reset
    temp[idx] = mtemp

for line in out:
    print(line)

# learn
print('Learning...')
i,j,v,u,w,bnorm,pmvol = np.transpose(out)

arg2 =  np.argwhere(bnorm-b0 < 0)[:,0]

temp = m_dipole.copy()
for j in np.argsort(arg2):
    v,u,w,m = data[j]
    idx = id3[j]

    temp[idx] = m * temp[idx]

    bnorm,pmvol = calc_bnorm(temp)
    #print(j,idx,u,v,w,bnorm,pmvol)
    out.append([j,idx,v,u,w,bnorm,pmvol])

for line in out:
    print(line)


### now write

pd =  mr.ReadFAMUS(f_famus)
# set up coordinates

N_dipoles = len(pd.X)
N_towers  = int(N_dipoles / N_layers)
rings = load_rings(pd)

# main loop
for j in arg2:
    v,u,w,m = data[j]

    idx = idx_3d(u,v,w)
    pd.pho[idx] = m
print('  making %i changes' % len(data) )

# write updated output
fout = f_famus[:-6] + '-{}v5.focus'.format( fin[:-4] )
pd.writefile(fout)

