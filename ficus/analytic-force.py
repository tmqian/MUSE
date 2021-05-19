import jax.numpy as np
from jax import grad, vmap


# Exact Analytic field from Cifta

def F(a,b,c):
    d = a*np.arcsinh( b/np.sqrt(a*a + c*c) )
    e = b*np.arcsinh( a/np.sqrt(b*b + c*c) )
    
    r = np.sqrt( a*a + b*b + c*c )
    f = c*np.arctan( (a*b)/(c*r) ) 

    
    return 2 * (d+e-f) # /np.sqrt(np.pi)

    
def V_local(r,s,q):
    
    x,y,z = r/s
    
    w = F(1-x, 1-y, z) + F(1-x, 1+y, z) + F(1+x, 1-y, z) + F(1+x, 1+y, z)
    return q/s * w / 8

def V_mag(r,H,L,M):
    x,y,z = r
    
    tx = np.heaviside(L/2 - np.abs(x),0.5)
    ty = np.heaviside(L/2 - np.abs(y),0.5)
    tz = np.heaviside(H/2 - np.abs(z),0.5)
    
    oz = np.heaviside(H/2 - z, 0.5) - np.heaviside(H/2 + z, 0.5)
    
    Br = M*4*np.pi/1e7 
    return - Br * (z*tz - oz*H/2) * tx*ty



def to_cartesian(r,zhat,xhat):

    yhat = np.cross(zhat,xhat)

    z = np.dot(r,zhat)
    y = np.dot(r,yhat)
    x = np.dot(r,xhat)

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


def V_general(r1,r0,n1,n2,H,L,M):


    n1hat = n1 / np.sqrt( np.dot(n1,n1) )
    n2hat = n2 / np.sqrt( np.dot(n2,n2) ) # this is the same for both plates (since there is no shear)

    rp = r1 - r0 - n1hat*(H/2)
    rm = r1 - r0 + n1hat*(H/2)

    Rp = to_cartesian(rp,n1hat,n2hat)
    Rm = to_cartesian(rm,n1hat,n2hat)

    # compute scalar potential
    Q = M * L * L / 1e7 # [A*m] a current potential K, times mu0/4pi
    S = L/2

    Vp = V_local(Rp,S,Q)
    Vm = V_local(Rm,S,Q)
    V0 = V_mag(r1-r0,H,L,M) # new

    return Vp - Vm + V0


def Vg_wrap(target,source):

    x1,y1,z1 = target
    x0,y0,z0,nx,ny,nz,ux,uy,uz, H,L,M = source

    r1 = np.array([x1,y1,z1])

    r0 = np.array([x0,y0,z0])
    n1 = np.array([nx,ny,nz])
    n2 = np.array([ux,uy,uz])

    return V_general(r1,r0,n1,n2,H,L,M)


def Bvec(target,source):

    gradV = vg2(target,source)
    B     = -1*np.sum(gradV, axis=0)
    return B


def split(A):
    Ax,Ay,Az = A.T
    Amag = np.sqrt( np.sum(A*A, axis=1) )
    return Ax,Ay,Az,Amag


## first loop target, then source
# this has the advantage of putting sources (which we want to sum over) on axis=0

# Calculate B vector field
vg1 = vmap( grad(Vg_wrap),(0,None))
vg2 = vmap( vg1,(None,0))



# Cacluate V scalar potential
vt1 = vmap( Vg_wrap,(0,None))
vt2 = vmap( vt1,(None,0))


L = 0.25*2.54/100 # quarter inch sq, as m**2
M = 1.1658e6 # units A/m
H = L

# useful for ref
Br = M * 4*np.pi/1e7
