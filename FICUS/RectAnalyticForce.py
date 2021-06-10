import jax.numpy as np
from jax import grad, vmap, jit

from FICUS import MagnetReader as mr

import time
import pdb
'''
    Exact Analytic field from Cifta
    This is a fork of AnalyticForce.py
    which attempts to generalize from Square to Rectangular Plates

    9 June 2021
'''

# enable 64 bit, which is necessary for JIT to get values right near the source
from jax.config import config
config.update("jax_enable_x64", True)


def F(a,b,c):
    d = a*np.arcsinh( b/np.sqrt(a*a + c*c) )
    e = b*np.arcsinh( a/np.sqrt(b*b + c*c) )
    
    r = np.sqrt( a*a + b*b + c*c )
    f = c*np.arctan( (a*b)/(c*r) ) 

    
    return 2 * (d+e-f) # / np.pi ## This factor of 1/pi cancels with a pi in V

    
def V_local(R,magnet):

    # load position and dimensions
    x,y,z  = R
    M,H,L,W = magnet
    
    # lengths are normalized x = X / (L/2)
    X,Y,Z = 2*R 
    x = X/L
    y = Y/W
    z = Z/H

    # for general (non-square) case, integrals must be rescaled
    a = L/H
    b = W/H

    #a = 1
    #b = 1

    # sigma has dimensions of M [A/m], since Q is [A m]
    Q = M * L * W / 1e7 # [A*m] a current potential K, times mu0/4pi
    #pdb.set_trace()
    w =  F( a*(1-x), b*(1-y), z) + F( a*(1-x), b*(1+y), z) \
       + F( a*(1+x), b*(1-y), z) + F( a*(1+x), b*(1+y), z)

    return M * H / 8 / 1e7 # * np.pi ## This factor of pi cancels with a 1/pi in F
    #return Q/L * w / 4 # * np.pi ## This factor of pi cancels with a 1/pi in F
    #return Q * w / 4 # * np.pi ## This factor of pi cancels with a 1/pi in F
    # just Q gave something in the Tesla range
    # also interesting to note that a=b seemed to give the same value for a != 1. Is this reasonable
    # okay good, they were only similar but not the same


# magnetization potential (homogeneous internal field)
# assumes primed coordinates, where x,y are normal to face, and z is parallel to magnetization
def V_mag_local(r,magnet):

    # load position and dimensions
    x,y,z  = r
    M,H,L,W = magnet

    # set up walls
    tx = np.heaviside(L/2 - np.abs(x),0.5)
    ty = np.heaviside(W/2 - np.abs(y),0.5)
    tz = np.heaviside(H/2 - np.abs(z),0.5)

    # magnetization gradient
    oz = np.heaviside(H/2 - z, 0.5) - np.heaviside(H/2 + z, 0.5)

    # remanent field [T]
    Br = M*4*np.pi/1e7
    return - Br * (z*tz - oz*H/2) * tx*ty

# assumes input is unit vectors (else projection is skewed)
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
    M is the magnetization of the material [A/m]
    H is height of magnet (separation of electrostatic plates) n1 direction
    L is the side length, defined by n2 direction
    W is the width, implicitly defined by n3 = n1 x n2 (= L for square case)
'''


def V_general(r1,r0,n1,n2,M,H,L,W):

    rp = r1 - r0 - n1*(H/2)
    rm = r1 - r0 + n1*(H/2)

    Rp = to_cartesian(rp     , n1, n2)
    Rm = to_cartesian(rm     , n1, n2)
    dR = to_cartesian(r1 - r0, n1, n2)

    # compute scalar potential
    magnet = np.array([M,H,L,W])
    #pdb.set_trace()
    Vp = V_local(Rp,magnet)
    Vm = V_local(Rm,magnet)
    V0 = V_mag_local(dR,magnet) 

    return Vp - Vm + V0

'''
   Assume H is parallel to n1, local z
          L is parallel to n2, local x
          W is parallel to n3 (implicitly defined as n1 x n2)

   Assume n1, n2, and m0 are unit vectors
'''
def Vg_wrap_3D(target,source):

    x1,y1,z1 = target
    x0,y0,z0, nx,ny,nz, ux,uy,uz, H,L,W, M, mx,my,mz = source # new convention
    #x0,y0,z0,nx,ny,nz,ux,uy,uz, H,L,M = source

    r1 = np.array([x1,y1,z1])

    r0 = np.array([x0,y0,z0])
    n1 = np.array([nx,ny,nz])
    n2 = np.array([ux,uy,uz])
    m0  = np.array([mx,my,mz]) 

    # go to local coordinates
    m1,m2,m3 = to_cartesian(M*m0, n1, n2)

    V1 = V_general(r1,r0,n1,n2, m1,H,L,W)
    V2 = V_general(r1,r0,n2,n3, m2,L,W,H)
    V3 = V_general(r1,r0,n3,n1, m3,W,H,L)
    return V1 + V2 + V3

# backward compatibility
def Vg_wrap(target,source):

    x1,y1,z1 = target
    x0,y0,z0,nx,ny,nz,ux,uy,uz, H,L,M = source
    #source = np.array([x0,y0,z0,nx,ny,nz,ux,uy,uz, M,H,L]).T # from MagnetReader file

    r1 = np.array([x1,y1,z1])

    r0 = np.array([x0,y0,z0])
    n1 = np.array([nx,ny,nz])
    n2 = np.array([ux,uy,uz])
    
    #Q = M * L * L / 1e7 # [A*m] a current potential K, times mu0/4pi
    return V_general(r1,r0,n1,n2, M,H,L,L)

## first loop target, then source
# this has the advantage of putting sources (which we want to sum over) on axis=0

# Calculate B vector field
vg1 = vmap( grad(Vg_wrap),(0,None))
vg2 = vmap( vg1,(None,0))


def Bvec(target,source):

    gradV = vg2(target,source)
    B = -1*gradV
    return B

jit_Bvec = jit(Bvec)


# Cacluate V scalar potential
vt1 = vmap( Vg_wrap,(0,None))
vt2 = vmap( vt1,(None,0))


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
    Ax = A[:,0]
    Ay = A[:,1]
    Az = A[:,2]
    Amag = np.linalg.norm(A, axis=1)
    return Ax,Ay,Az,Amag


def calc_B(targets,source, B_func=jit_Bvec, n_step=5000, _face=True):
    # takes arbitrary B_function, defaults to Cifja

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
