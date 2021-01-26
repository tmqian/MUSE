#!/usr/bin/env python
# coding: utf-8

# In[4]:


from coilpy import *
import jax.numpy as np
import numpy as nnp
import matplotlib.pyplot as plt
from jax import grad
from jax.ops import index, index_add, index_update
from jax import vmap, device_put, jacfwd, jacrev
from jax.experimental import loops
from jax.numpy import save as sv
import pyevtk
from numpy.lib import recfunctions as rfn


# In[5]:


#set up file paths
maindir='/u/achambl2/FOCUS_Nov/FOCUS/examples/MUSE/'
h5='focus_muse-nd.h5'
famuspath='/u/achambl2/FAMUS/examples/MUSE/'
famus='10_19_inputcoils.focus'
plas='coils_only.plasma'
plas_in=maindir+'ncsx_2p_vacuum.plasma'


# In[6]:


h5file=FOCUSHDF5(famuspath+h5)
h5foc=FOCUSHDF5(maindir+'focus_coils_only.h5')#'focus_muse-nd_10_19.h5')


# In[7]:


Ntheta=h5file.Nteta
Nzeta=h5file.Nzeta
print(Ntheta,Nzeta)


# In[8]:


#open new file
mags=Dipole.open(famuspath+'10_19_full_dipoles.focus')
print(len(mags.ox))


# In[9]:


#written so one can concatenate dipole list if shorter run-time is desired
length=len(mags.ox)
q=mags.momentq
ox=np.array(mags.ox[0:length])
oy=np.array(mags.oy[0:length])
oz=np.array(mags.oz[0:length])
pho=mags.pho[0:length]
mt=mags.mt[0:length]
mp=mags.mp[0:length]
mm=mags.mm[0:length]
rho=mags.rho[0:length]
print(length)


# In[10]:


xyz_dip=np.array((ox,oy,oz))


# In[11]:


#build vector of dipole moments
m0i=1
mx=(pho**q)*mm*np.sin(mt)*np.cos(mp)
my=(pho**q)*mm*np.sin(mt)*np.sin(mp)
mz=(pho**q)*mm*np.cos(mt)
moments=np.array((mx,my,mz))#dims=(Ndip,3)
mom_trans=np.transpose(moments)


# In[12]:


#setup grid of indices used to build position list
#these are the integer indices used to index over theta and zeta on the plasma surface
thetai=np.linspace(0, Ntheta-1,Ntheta,dtype=int)
zetaj=np.linspace(0, Nzeta-1,Nzeta,dtype=int)
zet,thet=np.meshgrid(zetaj,thetai, indexing='ij')


# In[13]:


#use theta zeta grid to extract Cartesian positions on plasma surface
posnx=np.ravel(np.array((h5file.xsurf[zet,thet])))
posny=np.ravel(np.array((h5file.ysurf[zet,thet])))
posnz=np.ravel(np.array((h5file.zsurf[zet,thet])))


# In[14]:


#build full position array (an Ndim list of all the 3dim positions on the plasma surface)
positions=np.transpose(np.array((posnx,posny,posnz)))
arr=np.arange(len(positions))
#ravel the dipole data for use in bfield fns
ravelxyz=np.ravel(np.transpose(xyz_dip))


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


# In[16]:


#test eval_pos to compare to CoilPy output
#point0=np.array((0,0,0))
#print(bfield_mut(ravelxyz, np.transpose(moments), point0))


# In[17]:


#compare w/ CoilPy
#print(mags.bfield(point0))


# In[18]:


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


# In[19]:


#Map both functions over the list of positions to compute bfield at every posn in the list
vmap_bfield=vmap(bfield,(None,None,0,None))
vmap_bfield_s=vmap(bfield_mut,(None,None,0))


# In[20]:


#Choose the number of mutable dipoles now. bfield_fixed is being defined here
# for later use in chi2b calc
bfield_fixed=vmap_bfield(ravelxyz,np.transpose(moments),positions,35166)
#print(bfield_fixed)


# In[21]:


bfield_muta=vmap_bfield_s(ravelxyz[:35166],np.transpose(moments),positions) #concatenate ravelxyz to consider only the mutable dipoles
#this subset from ravelxyz will be the group used in grad to compute the derivative
#we can use this method to calculate shape gradients and Hessian matrices for a subset of dipoles.
#must be the first ones in the list
#print(bfield_muta)


# In[22]:


bfield_all=bfield_muta+bfield_fixed
print(bfield_all)


# In[23]:


#extract components of normal field vector computed by FOCUS
nx=h5file.nx
ny=h5file.ny
nz=h5file.nz


# In[24]:


#isolate x, y, and z components of field computed above
bx=np.reshape(np.transpose(bfield_all)[0],(Nzeta,Ntheta))
by=np.reshape(np.transpose(bfield_all)[1],(Nzeta,Ntheta))
bz=np.reshape(np.transpose(bfield_all)[2],(Nzeta,Ntheta))


# In[25]:


#calc normal component of field: \textbf{B}_M \cdot \textbf{n}
BMdotN=bx*nx+by*ny+bz*nz
BdotN=BMdotN+h5foc.Bn


# In[26]:


#extract FOCUS's calculation of B dot n
Bn=h5file.Bn


# In[27]:


#sum to see if your calc agrees w/ FOCUS-- these #s should be almost equal
np.sum(BMdotN), np.sum(h5file.Bn)


# In[28]:


#see if your chi2b value is equivalent to the one output by FOCUS
chib=np.sum(BdotN*BdotN*h5file.nn*2*np.pi**2/(Ntheta*Nzeta))
chib_famus=np.sum(Bn*Bn*h5file.nn*2*np.pi**2/(Ntheta*Nzeta))


# In[29]:


chib, chib_famus, chib/chib_famus,h5file.evolution[3]


# In[30]:


def chi2b(ravel_xyz):
    bfield_mutable=vmap_bfield_s(ravel_xyz,mom_trans,positions)
    bfield_alls=np.transpose(bfield_fixed+bfield_mutable)
    bx=np.reshape(bfield_alls[0],(Ntheta,Nzeta))
    by=np.reshape(bfield_alls[1],(Ntheta,Nzeta))
    bz=np.reshape(bfield_alls[2],(Ntheta,Nzeta))
    BMdotn_all=bx*nx+by*ny+bz*nz
    Bdotn_all=BMdotn_all+np.array(h5foc.Bn)
    Bn_integral=np.sum(Bdotn_all*Bdotn_all*h5file.nn*2*np.pi**2/(Ntheta*Nzeta))
    return Bn_integral


# In[31]:


#make list of vectors
def vectors(i):
    vec=np.zeros(len(ravelxyz[:31566]))
    vec_jax=index_update(vec,i,np.array(int(1)))
    return vec_jax


# In[32]:


def hvp(f, x, i):
    Hi=grad(lambda x: np.vdot(grad(f)(x), vectors(i)))(x)
    #with open('Htest', 'wb') as f:
    #        np.save(f, Hi)
    return Hi


# In[59]:


for i in nnp.arange(0,11722):
    H=hvp(chi2b,ravelxyz[:31566],i)[i:]
    with open("HessianMUSE.npy", "rb+") as f:
            f.read()
            nnp.save(f, H,allow_pickle=False)


# In[58]:


#with open("HessianMUSE.npy","rb") as f:
#    a=nnp.load(f,allow_pickle=False)
#print(a)


# In[ ]:


#for i in nnp.arange(11723,23444):
#    H=hvp(chi2b,ravelxyz,i)
#    with open("HessianMUSE_23444.txt", "r+") as f:
#            f.read()
#            nnp.savetxt(f, H)


# In[ ]:


#for i in nnp.arange(23445,35166):
#    H=hvp(chi2b,ravelxyz,i)
#    with open("HessianMUSE_35166.txt", "r+") as f:
#            f.read()
#            nnp.savetxt(f, H)


# In[ ]:


#for i in nnp.arange(35167,46888):
#    H=hvp(chi2b,ravelxyz,i)
#    with open("HessianMUSE_46888.txt", "r+") as f:
#            f.read()
#            nnp.savetxt(f, H)


# In[ ]:


#for i in nnp.arange(46889,58610):
#    H=hvp(chi2b,ravelxyz,i)
#    with open("HessianMUSE_58610.txt", "r+") as f:
#            f.read()
#            nnp.savetxt(f, H)


# In[ ]:


#for i in nnp.arange(58611,70332):
#    H=hvp(chi2b,ravelxyz,i)
#    with open("HessianMUSE_70332.txt", "r+") as f:
#            f.read()
#            nnp.savetxt(f, H)


# In[ ]:


#for i in nnp.arange(70333,82054):
#    H=hvp(chi2b,ravelxyz,i)
#    with open("HessianMUSE_82054.txt", "r+") as f:
#            f.read()
#            nnp.savetxt(f, H)


# In[ ]:


#for i in nnp.arange(82055,93776):
#    H=hvp(chi2b,ravelxyz,i)
#    with open("HessianMUSE_93776.txt", "r+") as f:
#            f.read()
#            nnp.savetxt(f, H)


# In[ ]:


#for i in nnp.arange(93777,105498):
#    H=hvp(chi2b,ravelxyz,i)
#    with open("HessianMUSE_105498.txt", "r+") as f:
#            f.read()
#            nnp.savetxt(f, H)


# In[ ]:


#for i in nnp.arange(105499,117220):
#    H=hvp(chi2b,ravelxyz,i)
#    with open("HessianMUSE_117220.txt", "r+") as f:
#            f.read()
#            nnp.savetxt(f, H)


# In[ ]:


#for i in nnp.arange(117221,128942):
#    H=hvp(chi2b,ravelxyz,i)
#    with open("HessianMUSE_128942.txt", "r+") as f:
#            f.read()
#            nnp.savetxt(f, H)


# In[ ]:


#for i in nnp.arange(128943,140664):
#    H=hvp(chi2b,ravelxyz,i)
#    with open("HessianMUSE_140664.txt", "r+") as f:
#            f.read()
#            nnp.savetxt(f, H)

