import numpy as np

'''
    This program writes FAMUS seeds.

    Updated 1 February 2021
    Contact: tqian@pppl.gov
'''

# user input
R0 = 0.3048        # major radius
a0 = 0.105         # minor radius of (center of mass of) inner most magnet layer (smallest minor radius)
N_rings = 98       # number of poloidal rings
N_layers = 12      # number of radial layers

# define the finite volume of magnets
dz = 0.00635       # zeta  - toroidal
dt = 0.00635       # theta - poloidal
da = 0.00635/4     # a     - flux
mdip = 0.2985/4    # magnetization of one dipole


def uv_to_xyz(u,v, a=0.09, R=0.3048):

    xyz = []

    for j in np.arange( len(u) ):

        zeta  = u[j]
        theta = v[j]

        rho = R + a*np.cos(theta)
        x = rho*np.cos(zeta)
        y = rho*np.sin(zeta)
        z = a*np.sin(theta)

        xyz.append( np.array([x,y,z]) )

    return np.array(xyz) # m


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


# convert to spherical coordinates: theta-azimuth, phi-polar
def n_to_sphere(nhat):

    tp = []

    N = len(nhat)
    for k in np.arange(N):
        x,y,z = nhat[k]

        theta = np.arccos(z)
        phi = np.arctan2(y,x)

        tp.append( [theta,phi] )

    return np.transpose(tp)

# bridge
def xyz_to_sphere(xyz):

    nhat   = xyz_to_n(xyz)
    mt,mp = n_to_sphere(nhat)

    return mt, mp

# manipulating magnets
def norm(v):
    v = np.array(v)
    return v / mag(v)

def mag(v):
    return np.sqrt( np.sum(v*v) )


def get_effective_radius(t,a0):
    cost = np.cos(t)
    sint = np.sin(t)
    if (cost > 0): # outside
        R = R0 + a0*cost - dt/2*sint
    else:          # inside
        R = R0 + a0*cost + dt/2*sint
    return R


# calculates (equal) spacing for toroidal rings
#    given toroidal manget thickness dt, computes the space between magnets
#    if negative, the array overlaps and is non-physical
def calc_Tx(a0,N_theta):
    
    Tx = np.linspace(0,np.pi*2,N_theta, endpoint=False)
    Tx += Tx[1]/2

    min_circ = a0*2*np.pi # minimum circumference
    print('min poloidal space between magnets: %f' % (min_circ/N_theta - dt))
    
    return Tx

# makes a toroidal grid, for a given poloidal ring
#    contains midpoints and cell walls
def axZeta(t,a0=0.095,cell_wall=0.001):
    R = get_effective_radius(t,a0)
    C = 2*np.pi*R
    
    # toroidal spaces per quarter period
    midpoint_spacer = cell_wall*2               # spacer at pi/4 adding an extra double cell wall
    nz = int( (C/4 - midpoint_spacer) / (dz+cell_wall) / 2 ) # extra factor of 2 for half a half-pipe
    
    # only make space at 45 deg
    s = cell_wall/R   # convert to radians: (s/2pi) = cell_wall / 2pi R
    Zx1 = np.linspace(0, np.pi/4 - s, nz, endpoint=False)
    Zx1 += Zx1[1]/2 # half step -> zone-centers
    
    Zx2 = np.pi/2 - Zx1 # reflect
    
    Zx = np.concatenate((Zx1,Zx2))
    return Zx


def make_xyz(a0,nt,N_layers):

    # init
    Tx = calc_Tx(a0,nt)
    aN = a0 + da*(N_layers-1) # list magnet boundaries
    a_layers = np.linspace(a0,aN,N_layers) + (da/2) # half step into COM positions

    # get uv coordinates
    arr = []
    for t in Tx:

        Zx = axZeta( t )       # makes a toroidal grid, for a given poloidal ring
        for z in Zx:
            arr.append( [t,z] )

    # convert to points in 3D
    u,v = np.transpose(arr)
    xyz_layers = [ uv_to_xyz(v,u,a=t) for t in a_layers ]
    np.shape(xyz_layers)

    # format list
    N_surface = len(arr)
    N_dipoles = N_layers * N_surface
    xyz_volume = np.reshape(xyz_layers, (N_dipoles,3))

    return xyz_volume

def write_FAMUS_geometry(fname,xyz, M0,pho=1e-4, Lc=0,Ic=1, q=1,symm=2):
    
    # prep    
    mt,mp = xyz_to_sphere(xyz)    
    N = len(xyz) 
  
    # write
    with open(fname,'w') as f:  
        
        # start writing
        h1 = '# Total number of coils,  momentq \n'
        h2 = '   {}, {} \n'.format(N,q)
        h3 = '# coiltype, symmetry,  coilname,  ox,  oy,  oz,  Ic,  M_0,  pho,  Lc,  mp,  mt \n'
        f.write(h1)
        f.write(h2)
        f.write(h3)
        
        for j in np.arange(N):
            
            x,y,z = xyz[j]
            phi   = mp[j]  
            theta = mt[j]
            rho   = pho**q
            m0    = M0

            dname = 'pm{:08d}'.format(j)
            line = '{}, {}, {}, {:.6f}, {:.6f}, {:.6f}, {}, {:.3e}, {}, {}, {:.6f}, {:.6f} \n'.format(
                2,symm,dname,x,y,z,Ic,m0,rho,Lc,phi,theta)
            f.write(line)
    
    print('Wrote {} magnets to file {}'.format(N,fname))
    return N


### main
xyz = make_xyz(a0, N_rings, N_layers)
fname = 'a{}_p{}_n{}.focus'.format( int(a0*1000), N_rings, N_layers)
write_FAMUS_geometry(fname, xyz, mdip)

