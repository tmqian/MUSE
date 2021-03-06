import jax.numpy as np
from jax import grad, vmap, jit

from FICUS import MagnetReader as mr

import time
import pdb

### Exact Analytic field from Cifta
# updated 26 May 2021


# enable 64 bit, which is necessary for JIT to get values right near the source
from jax.config import config
config.update("jax_enable_x64", True)


def F(a,b,c):
    d = a*np.arcsinh( b/np.sqrt(a*a + c*c) )
    e = b*np.arcsinh( a/np.sqrt(b*b + c*c) )
    
    r = np.sqrt( a*a + b*b + c*c )
    f = c*np.arctan( (a*b)/(c*r) ) 

    
    return 2 * (d+e-f) # /np.sqrt(np.pi)

    
def V_local(R,magnet):
    
    X,Y,Z = 2*R
    M,H,L,W = magnet.T

    x = X/L
    y = Y/W
    z = Z/H
    
    a = L/H
    b = W/H

    w =  F( a*(1-x), b*(1-y), z) + F( a*(1-x), b*(1+y), z) \
       + F( a*(1+x), b*(1-y), z) + F( a*(1+x), b*(1+y), z)

    return (M*H/1e7) * w / 4


# magnetization potential (homogeneous internal field)
# assumes primed coordinates, where x,y are normal to face, and z is parallel to magnetization
def V_mag_local(r,magnet):

    # load position and dimensions
    x,y,z  = r
    M,H,L,W = magnet.T

    # set up walls
    tx = np.heaviside(L/2 - np.abs(x),0.5)
    ty = np.heaviside(W/2 - np.abs(y),0.5)
    tz = np.heaviside(H/2 - np.abs(z),0.5)

    # magnetization gradient
    oz = np.heaviside(H/2 - z, 0.5) - np.heaviside(H/2 + z, 0.5)

    # remanent field [T]
    Br = M*4*np.pi/1e7
    return - Br * (z*tz - oz*H/2) * tx*ty


# returns xyz, where n1 is zhat and n2 is xhat
def to_cartesian(r,zhat,xhat):

    yhat = np.cross(zhat,xhat)
 
    x = np.dot(r,xhat)
    y = np.dot(r,yhat)
    z = np.dot(r,zhat)

    return np.array([x,y,z])



'''
    r1 is test charge location
    r0 is COM of source magnet
    n1 is orientation of magnet (not necessarily normalized)
    n2 is 2nd orientation of magnet (direction tangent to one rectangular face, also not necessarily normalized)
    H is height of magnet (separation of electrostatic plates)
    L is the side length of face of magnet (assumed to be SQ)
    M is the magnetization of the material [A/m]
'''


def V_general(r1,r0,n1,n2,H,L,W,M):

    rp = r1 - r0 - n1*(H/2)
    rm = r1 - r0 + n1*(H/2)

    Rp = to_cartesian(rp,n1,n2)
    Rm = to_cartesian(rm,n1,n2)
    dr = to_cartesian(r1 - r0,n1,n2)

    # compute scalar potential
    magnet = np.array([M,H,L,W])
    Vp = V_local(Rp,magnet)
    Vm = V_local(Rm,magnet)
    V0 = V_mag_local(dr,magnet) 

    return Vp - Vm + V0


def Vg_wrap(target,source):

    x1,y1,z1 = target
    x0,y0,z0,nx,ny,nz,ux,uy,uz, H,L,M = source

    r1 = np.array([x1,y1,z1])

    r0 = np.array([x0,y0,z0])
    n1 = np.array([nx,ny,nz])
    n2 = np.array([ux,uy,uz])

    # for now W=L
    return V_general(r1,r0,n1,n2,H,L,L,M)

'''
   Assume H is parallel to n1, local z
          L is parallel to n2, local x
          W is parallel to n3 (implicitly defined as n1 x n2)

   Assume n1, n2, and m0 are unit vectors
'''
def Vg_wrap_3D(target,source):

    x1,y1,z1 = target
    try:
        x0,y0,z0, nx,ny,nz, ux,uy,uz, H,L, M, mx,my,mz = source # new convention
        W = L
        print('imported 15 variables, assuming W=L')
    except:
        x0,y0,z0, nx,ny,nz, ux,uy,uz, H,L,W, M, mx,my,mz = source # new convention
        print('imported 16 variables, with assuming W != L')
    #x0,y0,z0,nx,ny,nz,ux,uy,uz, H,L,M = source

    r1 = np.array([x1,y1,z1])

    r0 = np.array([x0,y0,z0])
    m0 = np.array([mx,my,mz]) 
    n1 = np.array([nx,ny,nz])
    n2 = np.array([ux,uy,uz])
    n3 = np.cross(n1,n2,axis=-1)

    # go to local coordinates
    m2,m3,m1 = to_cartesian(M*m0, n1, n2) + 1e-8
    # returns xyz, where n1 is zhat and n2 is xhat
    pdb.set_trace()

    V1 = V_general(r1,r0,n1,n2, m1,H,L,W)
    V2 = V_general(r1,r0,n2,n3, m2,L,W,H)
    V3 = V_general(r1,r0,n3,n1, m3,W,H,L)
    return V1 + V2 + V3

## first loop target, then source
# this has the advantage of putting sources (which we want to sum over) on axis=0

# Calculate B vector field
# new set for Bvec3 where M is not necessarily parallel to axis
vg3_1 = vmap( grad(Vg_wrap_3D),(0,None))
vg3_2 = vmap( vg3_1,(None,0))


def Bvec3(target,source):

    gradV = vg3_2(target,source)
    #pdb.set_trace()
    #foo = mask_self_interactions(8,len(source) )
    #gradV = gradV*foo
    #B     = -1*np.sum(gradV, axis=0)
    B = -1*gradV
    return B

jit_Bvec3 = jit(Bvec3)

# Calculate B vector field
vg1 = vmap( grad(Vg_wrap),(0,None))
vg2 = vmap( vg1,(None,0))

def Bvec(target,source):

    gradV = vg2(target,source)
    #pdb.set_trace()
    #foo = mask_self_interactions(8,len(source) )
    #gradV = gradV*foo
    #B     = -1*np.sum(gradV, axis=0)
    B = -1*gradV
    return B

jit_Bvec = jit(Bvec)

# Cacluate V scalar potential
vt1 = vmap( Vg_wrap,(0,None))
vt2 = vmap( vt1,(None,0))


#L = 0.25*2.54/100 # quarter inch sq, as m**2
#M = 1.1658e6 # units A/m
#H = L
#
## useful for ref
#Br = M * 4*np.pi/1e7

'''
    Dipole potential
    r1 is test charge location
    r0 is COM of source magnet
    n1 is orientation of magnet (not necessarily normalized)
    H is height of magnet (separation of electrostatic plates)
    L is the side length of face of magnet (assumed to be SQ)
    M is the magnetization of the material [A/m]
'''


def V_dipole(r1,r0,n1,H,L,M):

    nhat = n1 / np.sqrt( np.dot(n1,n1) )
    r = r1 - r0 

    # compute scalar potential
    m = nhat * M * (L*L*H)              # [A*m*m] dipole moment
    r2 = np.dot(r,r)
    rhat = r / np.sqrt(r2)
    V = np.dot(m,rhat) / r2 / 1e7       # times mu0/4pi

    return V


def Vd_wrap(target,source):

    x1,y1,z1 = target
    x0,y0,z0,nx,ny,nz, H,L,M = source

    r1 = np.array([x1,y1,z1])

    r0 = np.array([x0,y0,z0])
    n1 = np.array([nx,ny,nz])

    return V_dipole(r1,r0,n1,H,L,M)

# Calculate B vector field
vd1 = vmap( grad(Vd_wrap),(0,None))
vd2 = vmap( vd1,(None,0))

from scipy.linalg import block_diag

def mask_self_interactions(m,n):

    a = np.array([np.ones(m)]).T
    d = block_diag(*([a] * n))
    return 1-d



def Bvec_dipole(target,source):

    gradV = vd2(target,source)
    B     = -1*np.sum(gradV, axis=0)
    return B

jit_Bvec_dipole = jit(Bvec_dipole)


### define helper functions

def split(A):
    #t = Timer()
    #t.start('transpose')
    #Ax,Ay,Az = A.T
    Ax = A[:,0]
    Ay = A[:,1]
    Az = A[:,2]
    #t.stop()
    #Amag = np.sqrt( np.sum(A*A, axis=1) )
    #t.start('norm')
    Amag = np.linalg.norm(A, axis=1)
    #t.stop()
    return Ax,Ay,Az,Amag


def calc_B(targets,source, B_func=jit_Bvec, n_step=5000, _face=True):
    # takes arbitrary B_function, defaults to Cifja

    #N_steps = int(targets.shape[0]/n_step) + 1 


    t = Timer()
    t.start('B calc')
    print('  source shape:', source.shape)
    print('  target shape:', targets.shape)
    if (_face):
        n_step = int(len(source)/2)
        N_steps = int(targets.shape[0]/n_step) 
    else:
        N_steps = int(targets.shape[0]/n_step) + 1 # 6/4 this +1 might break the concatenation
    arr = np.arange(N_steps)*n_step
    
    Bout = [ B_func(targets[j:j+n_step], source) for j in arr]


    block = np.concatenate(Bout,axis=1)
    Btot = np.sum( block, axis=0).block_until_ready()
    t.stop()

    #print('bout_max', np.max( np.linalg.norm( np.array(Bout), axis=-1 ) ) ) # causes problems when Bout is irregular (targets len is not multiple of source len)
    print('block_max', np.max( np.linalg.norm( block , axis=-1 ) ) )
    print('btot_max', np.max( np.linalg.norm( Btot , axis=-1 ) ) )

    return Btot

def write_data(data,fout):
    
    print('Preparing to write file')
    
    with open(fout,'w') as f:
        f.write('X [m], Y[m], Z[m], Bx[T], By[T], Bz[T] \n')
        for line in data:
            x,y,z,bx,by,bz = line
            out = '{:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}'.format(x,y,z,bx,by,bz)
            print(out,file=f)
    print('  Wrote to %s' % fout)



### Timer function

class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self, msg=""):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        print('Start',msg)
        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"  Elapsed time: {elapsed_time:0.4f} seconds")
