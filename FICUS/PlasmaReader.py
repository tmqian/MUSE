import numpy as np
from FICUS import MagnetReader as mr
from coilpy import FOCUSHDF5

'''
   This class loads the Plasma Surface from a FAMUS .h5 file
   + r_plasma (list of 3-tuple points)
   + n_plasma (list of 3-tuple points)
'''

class ReadPlasma():


    def __init__(self,f_coil):

        try:
            h5_coilsurf = FOCUSHDF5(f_coil)
        except:
            print('attempting default')
            f_coil = 'tf-coils-pg19-halfp.h5'
            h5_coilsurf = FOCUSHDF5(f_coil)

        ### load plasma surface

        # use theta zeta grid to extract Cartesian positions on plasma surface
        px = np.ravel(h5_coilsurf.xsurf)
        py = np.ravel(h5_coilsurf.ysurf)
        pz = np.ravel(h5_coilsurf.zsurf)
        
        # extract components of normal field vector computed by FOCUS
        nx = np.ravel(h5_coilsurf.nx)
        ny = np.ravel(h5_coilsurf.ny)
        nz = np.ravel(h5_coilsurf.nz)
        
        # Load coil field
        Bn_coil  = np.ravel(h5_coilsurf.Bn)
        jacobian = np.ravel(h5_coilsurf.nn)
        
        # apply symmetry to coil field
        x = Bn_coil
        coil_symm = np.concatenate((x,-x,x,-x))
        
        # apply stellarator symmetry to plasma surface
        px,py,pz,coil_symm = mr.stellarator_symmetry(px,py,pz,Bn_coil)
        nx,ny,nz,jacob_symm = mr.stellarator_symmetry(nx,ny,nz,jacobian)
        
        r_plasma = np.transpose([px,py,pz])
        n_plasma = np.transpose([nx,ny,nz])
    
        N_plasma = len(r_plasma)
    
        # save to class
        self.r_plasma = np.transpose([px,py,pz])
        self.n_plasma = np.transpose([nx,ny,nz])
    
        self.coil_symm  = coil_symm
        self.jacob_symm = jacob_symm
    
        self.N_theta = h5_coilsurf.Nteta
        self.N_zeta  = h5_coilsurf.Nzeta
    
        self.N_plasma = N_plasma
    
        print('Loading Complete')
        print('  N_surface:', N_plasma)
   

    def calc_bnorm(self, B_vec):
    ### the shape of dipole_symm much match that of preloaded coil_symm on plasma surface

        dipole_symm = np.sum(B_vec * self.n_plasma, axis=1) # dot product

        # load values
        coil_symm  = self.coil_symm
        jacob_symm = self.jacob_symm
        N_theta    = self.N_theta
        N_zeta     = self.N_zeta

        # integrate residual error field (assumes nfp=2)
        Bn_symm = coil_symm + dipole_symm
        bnorm = np.sum(Bn_symm*Bn_symm * np.abs(jacob_symm)) * np.pi**2/(N_theta*4*N_zeta)/2 

        return bnorm
