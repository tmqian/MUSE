#!/usr/bin/env python
# coding: utf-8

# In[15]:


from coilpy import *
import jax.numpy as np
import numpy as nnp
import matplotlib.pyplot as plt
from jax import grad
from jax.ops import index, index_add, index_update
from jax import vmap, device_put, jacfwd, jacrev, jvp,vjp
from jax.experimental import loops
import pyevtk
from numpy.lib import recfunctions as rfn


# In[16]:


#set up file paths
maindir='/u/achambl2/FOCUS_Nov/FOCUS/examples/PM4Stell/'#path to FOCUS directory where Bn contribution from coils has been calculated
h5='focus_test.h5' #h5 file from FAMUS run with full torus of dipoles
famuspath='/u/achambl2/FAMUS/examples/PM4Stell/' #path to FAMUS directory where h5 and dipole files are located
famus_full='test_full_dipoles.focus' #full-period of dipoles


# In[17]:


h5file=FOCUSHDF5(famuspath+h5) #read h5 file


# In[38]:


h5foc=FOCUSHDF5(maindir+'focus_test.h5')


# In[18]:


#open new file
mags=Dipole.open(famuspath+famus_full) #read full torus of dipoles


# In[19]:


#written so one can shorten dipole list for debugging
length=len(mags.ox) #change length to shorten dipole list if necessary
q=mags.momentq
ox=np.array(mags.ox[0:length])
oy=np.array(mags.oy[0:length])
oz=np.array(mags.oz[0:length])
pho=mags.pho[0:length]
mt=mags.mt[0:length]
mp=mags.mp[0:length]
mm=mags.mm[0:length]
rho=mags.rho[0:length]


# In[20]:


xyz_dip=np.array((ox,oy,oz)) #build array of dipole position data
ravelxyz=np.ravel(np.transpose(xyz_dip))#ravel the dipole data for use in bfield fns


# In[32]:


#build vector of dipole moments
m0i=1
mx=(pho**q)*mm*np.sin(mt)*np.cos(mp)
my=(pho**q)*mm*np.sin(mt)*np.sin(mp)
mz=(pho**q)*mm*np.cos(mt)
moments=np.transpose(np.array((mx,my,mz)))#dims=(Ndip,3)


# In[22]:


Ntheta=h5file.Nteta
Nzeta=h5file.Nzeta


# In[23]:


#setup grid of indices used to build position list
#these are the integer indices used to index over theta and zeta on the plasma surface
thetai=np.linspace(0, Ntheta-1,Ntheta,dtype=int)
zetaj=np.linspace(0, Nzeta-1,Nzeta,dtype=int)
zet,thet=np.meshgrid(zetaj,thetai, indexing='ij')


# In[24]:


#use theta zeta grid to extract Cartesian positions on plasma surface
posnx=np.ravel(np.array((h5file.xsurf[zet,thet])))
posny=np.ravel(np.array((h5file.ysurf[zet,thet])))
posnz=np.ravel(np.array((h5file.zsurf[zet,thet])))


# In[25]:


#build full position array
positions=np.transpose(np.array((posnx,posny,posnz))) #dim=(Ntheta*Nzeta,3)


# In[26]:


#field calculation for mutable subset of dipoles
#ravel_xyz is the list of dipole positions. Truncate this list so only the first N mutable dipoles are run through the function as follows:
#bfield_mut(ravel_xyz[:N],...)
#note: due to in-function truncation, only the dipoles at the beginning of the list are permitted to be mutable.
#mut_moments are the complete list of moments for all dipoles (this is truncated inside the function to match the dipole position data)
#eval_pos is the evaluation position where the field is computed
def bfield_mut(ravel_xyz, mut_moments, eval_pos):
    n=int(len(ravel_xyz)/3) #for use to truncate moments
    oxyz=ravel_xyz.reshape(-1,3) #reshape dipole position data to have dim (N,3)
    listlen=int(len(np.transpose(oxyz)[0]))
    rxyz = oxyz - eval_pos #define r vector
    r=np.sqrt(np.sum(rxyz*rxyz,axis=1))#gives list of ri for i dipoles
    mxyz = np.transpose(np.array(np.transpose(mut_moments)))[:n] #truncate moments to fit dipoles
    rinv5=1/r**5
    rinv3=1/r**3
    dotprod=3*np.sum(np.multiply(mxyz,rxyz),axis=1)
    mdotroverr=np.multiply(dotprod,rinv5)
    term1=np.transpose(mdotroverr*np.transpose(rxyz))#this is ((3m * r)/|r|**5)*r
    term2=np.transpose(rinv3*np.transpose(mxyz))#this is m/r**3
    return 1E-7*np.sum(term1-term2,axis=0)


# In[27]:


#field calc for fixed dipoles
#ravel_xyz should be full list of all dipole data
#to obtain the full field at all points on plasma surface, set N=idx
#(this means the truncation point of ravel_xyz input to bfield_mut should be the same index where
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


# In[28]:


#Map both functions over the list of positions to compute bfield at every posn in the list
vmap_bfield=vmap(bfield,(None,None,0,None))
vmap_bfield_s=vmap(bfield_mut,(None,None,0))


# In[29]:


Nhper=6 #number of half-periods in the full torus
N=int(len(mags.ox)/Nhper) #N here is the number of dipoles in a half period
#can change N if you want to compute Hessian or shape gradient for more/fewer dipoles
Ndof=N*3


# In[43]:


bfield_muta=vmap_bfield_s(ravelxyz[:Ndof],moments,positions)
#truncate ravelxyz using Ndof to consider only the mutable dipoles
#this subset from ravelxyz will be the group used in grad and Hessian to compute the derivatives
#we can use this method to calculate shape gradients and Hessian matrices for a subset of dipoles.
#must be the first ones in the list


# In[45]:


bfield_all=bfield_muta+bfield_fixed


# In[46]:


point0=nnp.array((0,0,0))
mags.bfield(point0),bfield(ravelxyz,np.transpose(moments),point0,0)


# In[44]:


#bfield_fixed is being defined here as a variable for later use in chi2b calc
bfield_fixed=vmap_bfield(ravelxyz,moments,positions,Ndof)


# In[47]:


#isolate x, y, and z components of field computed above
bx=np.reshape(np.transpose(bfield_all)[0],(Nzeta,Ntheta))
by=np.reshape(np.transpose(bfield_all)[1],(Nzeta,Ntheta))
bz=np.reshape(np.transpose(bfield_all)[2],(Nzeta,Ntheta))


# In[48]:


#extract components of normal field vector computed by FOCUS
nx=h5file.nx
ny=h5file.ny
nz=h5file.nz


# In[49]:


#calc normal component of field: \textbf{B}_M \cdot \textbf{n}
#BM=np.reshape(np.sum(bfield_all*(n/nn[:,np.newaxis]),axis=1),(Ntheta,Nzeta))
#BMdotN=bx*normx+by*normy+bz*normz
BMdotN=bx*nx+by*ny+bz*nz
BdotN=BMdotN+h5foc.Bn
#np.shape(BM)


# In[50]:


#extract FOCUS's calculation of B dot n
Bn=h5file.Bn


# In[51]:


#sum to see if your calc agrees w/ FOCUS-- these #s should be almost equal
np.sum(BdotN), np.sum(h5file.Bn)


# In[52]:


#see if your chi2b value is equivalent to the one output by FOCUS
chib=np.sum(BdotN*BdotN*h5file.nn*2.*np.pi**2/(Ntheta*Nzeta))
chib_famus=np.sum(Bn*Bn*h5file.nn*2.*np.pi**2/(Ntheta*Nzeta))


# In[53]:


chib, chib_famus, h5file.evolution[3], chib/chib_famus, h5file.evolution[3]/chib_famus


# In[56]:


evaluation_factor=2 #set by the symmetry of the dipoles in FAMUS
def chi2b(ravel_xyz):
    bfield_mutable=vmap_bfield_s(ravel_xyz,moments,positions)
    bfield_alls=bfield_fixed+bfield_mutable
    bx=np.reshape(np.transpose(bfield_alls)[0],(Ntheta,Nzeta))
    by=np.reshape(np.transpose(bfield_alls)[1],(Ntheta,Nzeta))
    bz=np.reshape(np.transpose(bfield_alls)[2],(Ntheta,Nzeta))
    BMdotn_all=bx*nx+by*ny+bz*nz
    Bdotn_all=BMdotn_all+h5foc.Bn
    Bn_integral=np.sum(Bdotn_all*Bdotn_all*h5file.nn*evaluation_factor*np.pi**2/(Ntheta*Nzeta))
    return Bn_integral


# In[57]:


chi2b(ravelxyz[:Ndof]) #be sure to truncate position data when calling chi2b


# In[58]:


gradnt=nnp.array(grad(chi2b)(ravelxyz[:Ndof]))


# In[37]:


SGx=nnp.array(gradnt[::3])
SGy=nnp.array(gradnt[1::3])
SGz=nnp.array(gradnt[2::3])


# In[59]:


#famus='trial58c_bin.focus' #half period of dipole data
#magshp=Dipole.open(famuspath+famus) #use if you'd like a vtk file of the shape gradient


# In[38]:


#magshp.toVTK('PM4Stell_SG',SG=(SGx,SGy,SGz)) #write vtk shape gradient file


# In[60]:


def vectors(i): #make a single row in an NdofxNdof identity matrix
    vec=np.zeros(Ndof)
    vec_jax=index_update(vec,i,np.array(int(1)))
    return vec_jax


# In[28]:


def hvp(f, x, i): #hessian-vector product function-- computes ith row of the Hessian
    Hi=grad(lambda x: np.vdot(grad(f)(x), vectors(i)))(x)
    #with open('Htest', 'wb') as f:
    #        np.save(f, Hi)
    return Hi


# In[31]:


Nrows=Ndof #change this to compute some desired number of rows of the Hessian-- this will allow you to break up the calculation so as to not exceed memory or run-time.
open("row_index_config_Nrows.npy","wb").close() #initialize binary file to tell you what row the calculation stopped on if runtime is exceeded
open('config_Nrows.npy','wb').close() #initialize binary file for Hessian data
#I title my files with 'config' as the name of the dipole configuration, and with 'N
with open("config_Nrows.npy", "rb+") as f:
    for i in nnp.arange(0,Nrows):
        H=nnp.array(hvp(chi2b,ravelxyz[:Ndof],i)[i:]) #save Hessian row-by-row
        H.tofile(f)
        with open("row_index_config_Nrows.npy","rb+") as g:
                idx=nnp.array((i,i)) #for some reason errors appear if this array is 1D, so I'm writing the index value twice
                idx.tofile(g) #save row index number

