#!/usr/bin/env python
# coding: utf-8

# Updated 3 February


# In[4]:

from coilpy import *
import jax.numpy as np
import numpy as nnp
import matplotlib.pyplot as plt

from jax import grad,vmap, device_put, jacfwd, jacrev
from jax.ops import index, index_add, index_update
from jax.experimental import loops
from jax.numpy import save as sv

import sys
from numpy.lib import recfunctions as rfn
import MagnetReader as mr

_debug = False
_write = False

# In[5]:

f_coil  = 'tf-coils-pg19-halfp.h5'
h5_coilsurf = FOCUSHDF5(f_coil)

# In[7]:

try:
    f_famus = sys.argv[1]
except:
    f_famus = 'l14A_skim.focus'

#open new file
print('loading famus file: ', f_famus)
fd = mr.ReadFAMUS(f_famus)
print('  read N dipoles:', len(fd.X) )

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
length = fd.Nmag
q = fd.q

# In[10]:



#build vector of dipole moments

ox,oy,oz,pho = mr.stellarator_symmetry(ox,oy,oz,pho)
xyz_dip=np.array((ox,oy,oz))
rho = mm[0]*pho**q

print('  after symmetry N dipoles:', len(ox) )

mx = np.sin(mt)*np.cos(mp)
my = np.sin(mt)*np.sin(mp)
mz = np.cos(mt)
mx,my,mz,Mm = mr.stellarator_symmetry(mx,my,mz,mm)
mx = mx*rho
my = my*rho
mz = mz*rho
moments=np.array((mx,my,mz))#dims=(Ndip,3)
mom_trans=np.transpose(moments)


# print pmvol
pmvol = np.sum( np.abs(rho) )
print('pmvol:', pmvol)

# print dpbin
dpbin = np.sum( np.abs(pho) * (1 - np.abs(pho)) )
print('dpbin:', dpbin)


# In[12]:


#setup grid of indices used to build position list
#these are the integer indices used to index over theta and zeta on the plasma surface
Ntheta = h5_coilsurf.Nteta
Nzeta  = h5_coilsurf.Nzeta
print('Grid Ntheta, Nzeta:', Ntheta,Nzeta)
thetai = np.arange(Ntheta)
zetaj  = np.arange(Nzeta)
zet,thet = np.meshgrid(zetaj,thetai, indexing='ij')


#use theta zeta grid to extract Cartesian positions on plasma surface
posnx = np.ravel(np.array( (h5_coilsurf.xsurf[zet,thet]) ))
posny = np.ravel(np.array( (h5_coilsurf.ysurf[zet,thet]) ))
posnz = np.ravel(np.array( (h5_coilsurf.zsurf[zet,thet]) ))


# In[14]:


#build full position array (an Ndim list of all the 3dim positions on the plasma surface)
positions = np.transpose(np.array((posnx,posny,posnz)))

#ravel the dipole data for use in bfield fns
ravelxyz  = np.ravel(np.transpose(xyz_dip))


# In[15]:

#field calculation for mutable subset of dipoles
#ravel_xyz is the list of dipole positions. Concatenate this list so only 
#the first N mutable dipoles are run through the function as follows:
#bfield_mut(ravel_xyz[:N],...)
#note: due to in-function concatenation, only the dipoles at the beginning of the list are permitted to be mutable.
#mut_moments are the complete list of moments for all dipoles (this is concatenated inside the function)
#eval_pos is the evaluation position for which the field is computed
def bfield_mut(ravel_xyz, mut_moments, eval_pos):
    n=int(len(ravel_xyz)/3) #for use to concatenate moments later
    oxyz=ravel_xyz.reshape(-1,3)#reshape dipole position data
    listlen=int(len(np.transpose(oxyz)[0]))
    rxyz = oxyz - eval_pos
    r=np.sqrt(np.sum(rxyz*rxyz,axis=1))#gives list of ri for i dipoles
    mxyz = np.transpose(np.array(np.transpose(mut_moments)))[:n] #concatenate moments to fit dipoles
    rinv5=1/r**5
    rinv3=1/r**3
    dotprod=3*np.sum(np.multiply(mxyz,rxyz),axis=1)
    mdotroverr=np.multiply(dotprod,rinv5)
    term1=np.transpose(mdotroverr*np.transpose(rxyz))#this is ((3m * r)/|r|**5)*r
    term2=np.transpose(rinv3*np.transpose(mxyz))
    return 1E-7*np.sum(term1-term2,axis=0)



# In[18]:

# Bfield FIXED
# idx is the starting point
#field calc for fixed dipoles
#ravel_xyz should be full list of all dipole data
#to obtain the full field at all points on plasma surface, set N=idx
#(this means the concatenation point of ravel_xyz input to bfield_mut should be the same index where
#the calculation of the field for fixed dipoles begins)
def bfield(ravel_xyz,moments_,eval_pos,idx):
    n=int(idx/3)
    oxyz=ravel_xyz.reshape(-1,3)[n:,:]
    listlen=int(len(np.transpose(oxyz)[0]))
    rxyz = oxyz - eval_pos
    r=np.sqrt(np.sum(rxyz*rxyz,axis=1))#gives list of ri for i dipoles
    mxyz = np.transpose(np.array(np.transpose(moments_)))[n:]
    rinv5=1/r**5
    rinv3=1/r**3
    dotprod=3*np.sum(np.multiply(mxyz,rxyz),axis=1)
    mdotroverr=np.multiply(dotprod,rinv5)
    term1=np.transpose(mdotroverr*np.transpose(rxyz))#this is ((3m * r)/|r|**5)*r
    term2=np.transpose(rinv3*np.transpose(mxyz))
    return 1E-7*np.sum(term1-term2,axis=0)


#Map both functions over the list of positions to compute bfield at every posn in the list
vmap_bfield=vmap(bfield,(None,None,0,None))
vmap_bfield_s=vmap(bfield_mut,(None,None,0))


# In[20]:


#Choose the number of mutable dipoles now. bfield_fixed is being defined here
# for later use in chi2b calc
print('starting field calculation')
Ndof = len(ox)*3
print('  Ndof:', Ndof)

split = 0 # Amelia's function for reducing computation, another option is Ndof, or Ndof/2
bfield_fixed = vmap_bfield( ravelxyz,np.transpose(moments),positions,split )
#print('bfield_fixed:', bfield_fixed)


# In[21]:


bfield_muta=vmap_bfield_s(ravelxyz[:split],np.transpose(moments),positions) #concatenate ravelxyz to consider only the mutable dipoles
#this subset from ravelxyz will be the group used in grad to compute the derivative
#we can use this method to calculate shape gradients and Hessian matrices for a subset of dipoles.
#must be the first ones in the list
#print('bfield_muta:', bfield_muta)

bfield_all=bfield_muta+bfield_fixed
#print('bfield_all:', bfield_all)


# In[23]:


#extract components of normal field vector computed by FOCUS
nx = h5_coilsurf.nx
ny = h5_coilsurf.ny
nz = h5_coilsurf.nz

# In[24]:

#isolate x, y, and z components of field computed above
bx = np.reshape(np.transpose(bfield_all)[0],(Nzeta,Ntheta))
by = np.reshape(np.transpose(bfield_all)[1],(Nzeta,Ntheta))
bz = np.reshape(np.transpose(bfield_all)[2],(Nzeta,Ntheta))


# In[25]:


#calc normal component of field: \textbf{B}_M \cdot \textbf{n}
BMdotN  = bx*nx + by*ny + bz*nz
Bn_coil = h5_coilsurf.Bn
BdotN   = BMdotN + Bn_coil

#print('finished BdotN', nnp.shape(BdotN))

# In[26]:

#sum to see if your calc agrees w/ FOCUS-- these #s should be almost equal
#np.sum(BMdotN), np.sum(Bn) # print


# In[28]:


#see if your chi2b value is equivalent to the one output by FOCUS
chib = np.sum(BdotN*BdotN*h5_coilsurf.nn * np.pi**2/(Ntheta*Nzeta)/2 )

print('chib:',chib)
#check
if (_debug):
    #extract FOCUS's calculation of B dot n
    f_h5    = 'full-torus.h5'   # this one can be retired
    h5file      = FOCUSHDF5(f_h5)
    Bn=h5file.Bn
    chib_famus=np.sum(Bn*Bn*h5file.nn * 2*np.pi**2/(Ntheta*Nzeta))
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.contourf(Bn)
    plt.title('target')
    
    plt.subplot(1,3,2)
    plt.contourf(BMdotN)
    plt.title('calculated')
    
    plt.subplot(1,3,3)
    plt.contourf(-h5_coilsurf.Bn)
    plt.title('coils')
    
    plt.draw()
    plt.show()

    print('chib_famus:',chib_famus)
    print('ratio:',  chib/chib_famus)
# In[29]:



# In[30]:


def chi2b(ravel_xyz):
    bfield_mutable=vmap_bfield_s(ravel_xyz,mom_trans,positions)
    bfield_alls=np.transpose(bfield_fixed+bfield_mutable)
    bx=np.reshape(bfield_alls[0],(Ntheta,Nzeta))
    by=np.reshape(bfield_alls[1],(Ntheta,Nzeta))
    bz=np.reshape(bfield_alls[2],(Ntheta,Nzeta))
    BMdotn_all=bx*nx+by*ny+bz*nz
    Bdotn_all=BMdotn_all+np.array(h5_coilsurf.Bn)
    Bn_integral=np.sum(Bdotn_all*Bdotn_all*h5file.nn*2*np.pi**2/(Ntheta*Nzeta))
    return Bn_integral


# In[31]:


#make list of vectors
def vectors(i):
    vec=np.zeros(len(ravelxyz[:Ndof]))
    vec_jax=index_update(vec,i,np.array(int(1)))
    return vec_jax


# In[32]:


def hvp(f, x, i):
    Hi=grad(lambda x: np.vdot(grad(f)(x), vectors(i)))(x)
    #with open('Htest', 'wb') as f:
    #        np.save(f, Hi)
    return Hi

print('  quick paste: {}, {}, {}'.format(chib, dpbin, pmvol))
print('finished')
sys.exit()


#### Starting FICUS

R0 = 0.3048 # major radius [m]

r = np.sqrt(ox*ox + oy*oy) - R0
a = np.sqrt(r*r + oz*oz)

# AD magick: returns df/dx1, df/dy1, df/dz1, df/dx2, ...
df = grad(chi2b)(ravelxyz)

dfx,dfy,dfz = nnp.reshape(df, (length,3)).T 
#dfx,dfy,dfz = nnp.reshape( df, (3,length) )

data = nnp.transpose([ox,oy,oz,dfx,dfy,dfz,pho,mp,mt])

if (_write):
    print('writing binary')
    with open('test.bin','wb') as f:
        for line in data:
            #print(line,file=f)
            nnp.array(line).tofile(f)
    
    #print('writing text')
    #with open('test.txt','w') as f:
    #    for line in data:
    #        print(line,file=f)
    #        #nnp.array(line).tofile(f)
    #

