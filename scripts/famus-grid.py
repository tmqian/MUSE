# makes a grid for FAMUS PM optimization
import numpy as np

N_zeta = 90
N_theta = 60

R = 5
a = 0.6
NFP = 5

ds = 0.1
N_layers = 10

pax = np.linspace(0, np.pi/NFP, N_zeta, endpoint=False)
tax = np.linspace(0, np.pi*2, N_theta , endpoint=False)

sax = np.arange(N_layers) * ds

xyz = []
for s in sax:
    for t in tax:
        for p in pax:

            z = (a+s)*np.sin(t)
            r = R + (a+s)*np.cos(t)

            x = r*np.cos(p)
            y = r*np.sin(p)

            xyz.append([x,y,z])

xyz = np.array(xyz)

def norm(v):
    v = np.array(v)
    return v / mag(v)

def mag(v):
    return np.sqrt( np.sum(v*v) )

# given list of xyz vectors, return nhat relative to a torus
def xyz_to_n(xyz,R0=5):

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

# this function writes FAMUS input
def export_famus_PM(fname,xyz, M0=1, Ic=1, Lc=0, pho=1e-4):

    nhat = xyz_to_n(xyz)

    # load data
    x0,y0,z0 = xyz.T
    nx,ny,nz = nhat.T


    # compute data
    nr = np.sqrt( nx**2 + ny**2)
    phi = np.arctan2(ny,nx)
    theta = np.pi/2 - np.arctan2(nz,nr)

    # supplementary data
    N_mag   = len(xyz)
    PM_name = np.arange(N_mag)

    m = np.ones(N_mag) * M0
    ic = np.ones(N_mag) * Ic
    lc = np.ones(N_mag) * Lc

    # write
    fout = fname + '.focus'
    print(' Writing to', fout)
    with open(fout,'w') as f:
        h1   = '# Total number of coils,  momentq'
        h2   = '# coiltype, symmetry,  coilname,  ox,  oy,  oz,  Ic,  M_0,  pho,  Lc,  mp,  mt'
        print(h1, file=f)
        print('  {}   {}'.format(N_mag, 1) , file=f)
        print(h2, file=f)
        for j in np.arange(N_mag):
            line = '{:6d}, {:6d}, pm{:07d}, {:6e}, {:6e}, {:6e}, {}, {:6e}, {}, {}, {:6e}, {:6e}' \
                .format(2, 2, PM_name[j], x0[j], y0[j], z0[j], ic[j], m[j], pho, lc[j], phi[j], theta[j])
            print(line, file=f)
    print('Wrote %i magnets' % N_mag)
    print('  new file:', fout)



#import matplotlib.pyplot as plt
#
#x,y,z = xyz.T
#plt.figure()
#
#plt.subplot(1,2,1)
#plt.plot(x,y,'.')
#plt.axis('equal')
#
#plt.subplot(1,2,2)
#plt.plot(np.sqrt(x*x+y*y),z,'.')
#plt.axis('equal')
#plt.show()
#
#import pdb
#pdb.set_trace()
            

fname = 'test-w7x'
export_famus_PM(fname, xyz, M0=1e5)

