import matplotlib.pyplot as plt
import numpy as np

from netCDF4 import Dataset
import sys

SCALE = 1 #2.90 # hard coded for JET
_plot = True

SAVE = True
N_zeta = 20
N_theta = 20

def get(f,key):
    return f.variables[key][:]


# Taking Fourier modes CMN, selects for flux surface s_idx
# select sine or cosine for array
# input toroidal and poloidal angle axis (tax, pax)
# outputs 2D array Z(p,t)
### used by VMEC class
def fourier2space(Cmn, tax,pax, xm,xn, s_idx=48, N_modes=61, sine=True):

    arr = []

    #global xm,xn
    N_modes = len(xm)
    for j in np.arange(N_modes):

        m = int( xm[j] )
        n = int( xn[j] )

        c = Cmn[s_idx,j]

        if (sine):
            A = [[ c * np.sin( m*p - n*t )  for t in tax] for p in pax ]
        else:
            A = [[ c * np.cos( m*p - n*t )  for t in tax] for p in pax ]

        arr.append(A)

    return np.sum(arr, axis=0)

class vmec():

    def __init__(self,fin):
        f = Dataset(fin, mode='r')

        # 0D array
        self.nfp         = get(f,'nfp')
        self.ns          = get(f,'ns')
        self.mnmax       = get(f,'mnmax')
        self.aminor      = get(f,'Aminor_p')

        # 1D array
        self.xm          = get(f,'xm')
        self.xn          = get(f,'xn')
        self.xm_nyq      = get(f,'xm_nyq')
        self.xn_nyq      = get(f,'xn_nyq')

        self.iotaf       = get(f,'iotaf')
        self.presf       = get(f,'presf')

        # 2D array
        self.rmnc        = get(f,'rmnc')
        self.zmns        = get(f,'zmns')
        self.lmns        = get(f,'lmns')
        self.bmnc        = get(f,'bmnc')
        self.bsupumnc    = get(f,'bsupumnc')
        self.bsupvmnc    = get(f,'bsupvmnc')


    # actually, this gets a poloidal cross section (at const toroidal angle phi)
    def get_surface(self, N, phi=0, s=-1):

         pax = np.linspace(0,np.pi*2,N) # poloidal
         tax = np.array([phi])          # toroidal

         # positions
         R2d = fourier2space(self.rmnc, tax,pax, self.xm, self.xn, sine=False, s_idx=s)
         Z2d = fourier2space(self.zmns, tax,pax, self.xm, self.xn, sine=True,  s_idx=s)

         # cartisian coordinates for flux surface
         R = R2d [:,0]
         Z = Z2d [:,0]

         return R,Z


    def get_surface_area(self, N_zeta=20,N_theta=8,surface=-1):

        # get points
        r_arr = []
        for p in np.linspace(0,np.pi*2,N_zeta):
            r,z = self.get_surface(N_theta,phi=p,s=surface)

            x = r*np.cos(p)
            y = r*np.sin(p)

            r_arr.append(np.transpose([x,y,z]))

        r_arr = np.transpose(r_arr)

        # get displacements
        X_arr, Y_arr, Z_arr = r_arr
        dXdu, dYdu, dZdu, dXdv, dYdv, dZdv = uv_space(X_arr,Y_arr,Z_arr)

        # get area
        dRdu = np.array([dXdu, dYdu, dZdu])
        dRdv = np.array([dXdv, dYdv, dZdv])
        # compute cross product and take norm
        dArea = np.linalg.norm( np.cross(dRdu, dRdv,axis=0),axis=0)

        return np.sum(dArea)

    def surface_area(self):

        A = []
        for s in np.arange(self.ns):
            a = self.get_surface_area(surface=s)
            A.append(a)

        return A



'''
    def get_Bmod(self,N, phi=0, s=-1):

        pax = np.linspace(0,np.pi*2,N)
        tax = np.array([phi])

        #Bmod = fourier2space(self.bmnc, tax,pax, self.xm, self.xn, sine=False, s_idx=s)
        Bmod = fourier2space(self.bmnc, tax,pax, self.xm_nyq, self.xn_nyq, sine=False, s_idx=s)
        return Bmod

    def get_Buv(self,N, phi=0, s=-1):

        pax = np.linspace(0,np.pi*2,N)
        tax = np.array([phi])

        #Bu = fourier2space(self.bsupumnc, tax,pax, self.xm, self.xn, sine=False, s_idx=s)
        #Bv = fourier2space(self.bsupvmnc, tax,pax, self.xm, self.xn, sine=False, s_idx=s)
        Bu = fourier2space(self.bsupumnc, tax,pax, self.xm_nyq, self.xn_nyq, sine=False, s_idx=s)
        Bv = fourier2space(self.bsupvmnc, tax,pax, self.xm_nyq, self.xn_nyq, sine=False, s_idx=s)
        return Bu,Bv

    def get_lambda(self,N, phi=0, s=-1):

        pax = np.linspace(0,np.pi*2,N)
        tax = np.array([phi])

        Lambda = fourier2space(self.lmns, tax,pax, self.xm, self.xn, sine=True, s_idx=s)
        return Lambda


    def straight_surface(self, N, phi=0, s=-1):

        pax = np.linspace(0,np.pi*2,N) # poloidal
        tax = np.array([phi])          # toroidal

        Lambda = self.get_lambda(N, phi=0, s=-1)
        theta = pax + Lambda[:,0]

        # positions
        R2d = fourier2space(self.rmnc, tax,theta, self.xm, self.xn, sine=False, s_idx=s)
        Z2d = fourier2space(self.zmns, tax,theta, self.xm, self.xn, sine=True,  s_idx=s)

        # cartisian coordinates for flux surface
        R = R2d [:,0]
        Z = Z2d [:,0]

        return R,Z
'''




# making B
def uv_space(X_arr,Y_arr,Z_arr):

    dXdu = np.roll(X_arr,-1,axis=0) - X_arr
    dYdu = np.roll(Y_arr,-1,axis=0) - Y_arr
    dZdu = np.roll(Z_arr,-1,axis=0) - Z_arr

    dXdv = np.roll(X_arr,-1,axis=1) - X_arr
    dYdv = np.roll(Y_arr,-1,axis=1) - Y_arr
    dZdv = np.roll(Z_arr,-1,axis=1) - Z_arr

    return dXdu, dYdu, dZdu, dXdv, dYdv, dZdv




### main
fin = sys.argv[1]
#fin = 'wout_zot80-m.00000.nc'
vf = vmec(fin)


N_surf = vf.ns


area = []
for s in np.arange(N_surf):
    a = vf.get_surface_area(surface=s, N_zeta=N_zeta, N_theta=N_theta) * SCALE**2
    area.append(a)


### write
ns = vf.ns-1
asq = vf.aminor**2
if SAVE:
    tag = fin[5:-3]
    fout = 'surf_%s.csv' % tag
    with open(fout,'w') as fc:
       
        print( 'N_zeta, N_theta: %f, %f' % (N_zeta, N_theta) )
        for j in np.arange(N_surf):
            print('%i, %f, %f, %f' % (j, j/ns,  area[j], area[j]/asq), file=fc)
    
    
