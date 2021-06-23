import jax.numpy as np
from jax import vmap, jit

from scipy import special as sp

import Timer
#import time

### Exact Analytic field from Cifta
# updated 21 June 2021


# enable 64 bit, which is necessary for JIT to get values right near the source
from jax.config import config
config.update("jax_enable_x64", True)


mu0 = 4*np.pi/1e7


####
# Elliptic Integral Work Around
N = 1000
ax = np.linspace(0,1,N,endpoint=False)
ek = sp.ellipk(ax)
ee = sp.ellipe(ax)

#fin = 'ellipse-1000.csv'
#with open(fin) as f:
#    datain = f.readlines()
#data = np.array([line.strip().split(',') for line in datain], float)
#ax,ek,ee = data.T

def ellipk(m):
    return np.interp(m,ax,ek)

def ellipe(m):
    return np.interp(m,ax,ee)
###

# evaluate coil fields, r is cylindrical radius
def B_local(R,a,I):

    # unpack coordinates
    x,y,z = R/a
    r = np.sqrt( x*x + y*y )

    # geometric coefficients
    Q = (1 + r)**2 + z*z
    P = (1 - r)**2 + z*z
    cz = (r*r + z*z - 1) / P
    cr = (r*r + z*z + 1) / P

    # Elliptic integrals
    m = 4*r/Q
    print(m)
    #K = sp.ellipk(m)
    #E = sp.ellipe(m)
    K = ellipk(m)
    E = ellipe(m)

    # fields
    B0 = mu0*I/2/a
    B1 = B0 / np.pi / np.sqrt(Q)
    
    Bz = B1 * (K - cz*E)
    Br = B1 * (cr*E - K)
 
#    if (r != 0):
#        Bx = Br * (x/r)
#        By = Br * (y/r)
#    else:
#        Bx = 0
#        By = 0

    return np.array( [Br,Br,Bz] )
    #return np.array( [Bx,By,Bz] )


def to_cartesian(r,zhat,xhat):

    yhat = np.cross(zhat,xhat)
 
    x = np.dot(r,xhat)
    y = np.dot(r,yhat)
    z = np.dot(r,zhat)

    return np.array([x,y,z])



'''
    r1 is test charge location [m]
    r0 is COM of source magnet [m]
    n1 is orientation of magnet (not necessarily normalized)
    I  is the current [A]
    a  is the circular coil radius [m]
'''


def B_general(r1,r0,n1,I,a):

    n1hat = n1 / np.sqrt( np.dot(n1,n1) )
    n2hat = np.array( [0,0,1] )

    r = r1 - r0 
    R = to_cartesian(r,n1hat,n2hat)

    # compute local field
    B = B_local(R,a,I)

    return B


def B_wrap(target,source):

    x1,y1,z1 = target
    x0,y0,z0,nx,ny,nz,I,a = source

    r1 = np.array([x1,y1,z1])
    r0 = np.array([x0,y0,z0])
    n1 = np.array([nx,ny,nz])

    return B_general(r1,r0,n1,I,a)


## first loop target, then source
# this has the advantage of putting sources (which we want to sum over) on axis=0

# Calculate B vector field
B_1 = vmap( B_wrap,(0,None) )
B_2 = vmap( B_1,   (None,0) )

def Bvec(target,source):

    B = B_2(target,source)
    return B

jit_Bvec = jit(Bvec)

def calc_B(targets,source, B_func=jit_Bvec, n_step=3000):
    # takes arbitrary B_function, defaults to Cifja

    #N_steps = int(targets.shape[0]/n_step) + 1 


    t = Timer.Timer()
    t.start('B calc coils')
    print('  source shape:', source.shape)
    print('  target shape:', targets.shape)

#    if (_face):
#        n_step = int(len(source)/2)
#        N_steps = int(targets.shape[0]/n_step) 
#    else:
#        N_steps = int(targets.shape[0]/n_step) + 1 # 6/4 this +1 might break the concatenation
    
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


