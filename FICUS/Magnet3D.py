import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc

try:
    from coilpy import *
except:
    print('  note: coilpy package unavailable')

'''
    This program updates MagnetReader.py, specifically for reading FICUS input
    where magnetization M is not necessarily parallel to the face normal n1

    Last Updated: 23 Aug 2021
'''

### new file format
'''
    This reads Doug's file format, where magnets are saved as 3D volumes.
    Each magnet is encoded by 8 points in space, representing the corners.
    The convention is n1,n2,n3,n4 followed by s1,s2,s3,s4, where the difference between faces
    is used to encode orientation. This assumes the PM is magnetized parallel to axis.
'''
# this class reads a 3D magnet specified by 24 variables
class Magnet_3D():

    def __init__(self,fname, R=0.3048):

        # read file
        with open(fname) as f:
            datain = f.readlines()
        data  = np.array([ line.strip().split(',') for line in datain[1:] ], float)
        print('Data size:', data.shape)
        self.data = data
        self.N_magnets = len(data)

        # unpack data
        n1x, n1y, n1z, n2x, n2y, n2z, n3x, n3y, n3z, n4x, n4y, n4z, \
        s1x, s1y, s1z, s2x, s2y, s2z, s3x, s3y, s3z, s4x, s4y, s4z = data.T

        # sort into vectors
        self.n1 = np.array([n1x,n1y,n1z]).T
        self.n2 = np.array([n2x,n2y,n2z]).T
        self.n3 = np.array([n3x,n3y,n3z]).T
        self.n4 = np.array([n4x,n4y,n4z]).T

        self.s1 = np.array([s1x,s1y,s1z]).T
        self.s2 = np.array([s2x,s2y,s2z]).T
        self.s3 = np.array([s3x,s3y,s3z]).T
        self.s4 = np.array([s4x,s4y,s4z]).T

        # find center of mass, for each face
        self.nc = (self.n1 + self.n2 + self.n3 + self.n4)/4
        self.sc = (self.s1 + self.s2 + self.s3 + self.s4)/4

        # center of mass for body
        self.com = (self.nc + self.sc)/2

        # go to (u,v,a) coordinates to find whether each magnet points out or in
        self.R = R
        nx,ny,nz = self.nc.T
        nr = np.sqrt(nx*nx + ny*ny) - R

        self.nu = np.arctan2(ny,nx)
        self.nv = np.arctan2(nz,nr)
        self.na = np.sqrt(nz*nz + nr*nr)

        sx,sy,sz = self.sc.T
        sr = np.sqrt(sx*sx + sy*sy) - R

        self.su = np.arctan2(sy,sx)
        self.sv = np.arctan2(sz,sr)
        self.sa = np.sqrt(sz*sz + sr*sr)

        # polarity North:1 points out from torus, South:-1 points into torus
        self.sgn = np.array( (self.na - self.sa) > 0, int )*2 - 1
        length = np.abs( (self.na - self.sa) / 0.0015875 )  # 1/16"
        self.len = np.array( length + 0.1, int )

        # define orientations for Cifta
        nvec = self.nc - self.sc # not normalized
        pvec = self.compute_phat()

        H = np.linalg.norm(nvec, axis=1)
        self.nvec = nvec / H[:,np.newaxis] #normalized
        self.pvec = pvec / np.linalg.norm(pvec, axis=1)[:,np.newaxis] #normalized
        
        self.H = H
        self.L = self.compute_lengths()
        self.M =  1.1658e6 * np.ones(self.N_magnets) # hard coded for N-52 (Br = 1.465, KJ Magnetics)

        #self.xhat, self.yhat = self.compute_xy()

    def compute_lengths(self):
        
        n12 = np.linalg.norm(self.n1 - self.n2, axis=1)
        n23 = np.linalg.norm(self.n2 - self.n3, axis=1)
        n34 = np.linalg.norm(self.n3 - self.n4, axis=1)
        n41 = np.linalg.norm(self.n4 - self.n1, axis=1)
        s12 = np.linalg.norm(self.s1 - self.s2, axis=1)
        s23 = np.linalg.norm(self.s2 - self.s3, axis=1)
        s34 = np.linalg.norm(self.s3 - self.s4, axis=1)
        s41 = np.linalg.norm(self.s4 - self.s1, axis=1)

        L = np.mean( np.array([n12,n23,n34,n41,s12,s23,s34,s41]), axis=0)
        return L
    
    def compute_phat(self):
        
        x,y,z = self.com.T
        phi = np.array([-y,x,z*0]).T
        return phi 

    # local grid coordinates 
    # (!!) assumes C-shape 1-2-3-4 definition
    def compute_xy(self):

        # normalized to 1
        xhat = (self.n2 - self.n3 + self.s2 - self.s3
                              + self.n1 - self.n4 + self.s1 - self.s4) / 8

        yhat = (self.n2 - self.n1 + self.s2 - self.s1
                              + self.n3 - self.n4 + self.s3 - self.s4) / 8

        # normalized to sidelength L/2
        Xhat = xhat * self.L[:,np.newaxis] / 2
        Yhat = yhat * self.L[:,np.newaxis] / 2

        return Xhat, Yhat

    # for the Ciftja force
    def export_source(self):
        '''
        exports source array for CIFTJA force calculation
        [x0,y0,z0,nx,ny,nz,ux,uy,uz, H,L,M]
        '''

        x0,y0,z0 = self.com.T
        nx,ny,nz = self.nvec.T
        ux,uy,uz = self.pvec.T
        H = self.H
        L = self.L
        M = self.M

        source = np.array([x0,y0,z0,nx,ny,nz,ux,uy,uz, H,L,M]).T
        return source

    #### WARNING this is in the non-3d file (should be deleted)
    def export_m3d(self):
        '''
        exports source array for generalized M (not necessarily parallel to axis)
        r0, n1, n2, H,L,M, m3

        H could be encoded in n1
        L could be encoded in n2
        M could be encoded in m3

        Does redundancy add value?
        I'll keep it for now
        '''

        x0,y0,z0 = self.com.T
        nx,ny,nz = self.nvec.T
        ux,uy,uz = self.pvec.T
        H = self.H
        L = self.L
        M = self.M
        mx,my,mz = self.nvec.T

        source = np.array([x0,y0,z0,nx,ny,nz,ux,uy,uz, H,L,M, mx,my,mz]).T
        return source

    def export_source_dipole(self):

        x0,y0,z0 = self.com.T
        nx,ny,nz = self.nvec.T
        H = self.H
        L = self.L
        M = self.M

        source = np.array([x0,y0,z0,nx,ny,nz, H,L,M]).T
        return source

    # dipole approximation: returns center of top and bottom face
    def export_target_2(self):

        x1,y1,z1 = self.nc.T
        x2,y2,z2 = self.sc.T

        X = np.concatenate([x1,x2])
        Y = np.concatenate([y1,y2])
        Z = np.concatenate([z1,z2])

        target = np.array([X,Y,Z]).T
        return target

    # octopole approximation: returns 8 corners (deprecated)
    def export_corners(self):

        x1,y1,z1 = self.n1.T
        x2,y2,z2 = self.n2.T
        x3,y3,z3 = self.n3.T
        x4,y4,z4 = self.n4.T
        x5,y5,z5 = self.s1.T
        x6,y6,z6 = self.s2.T
        x7,y7,z7 = self.s3.T
        x8,y8,z8 = self.s4.T

        X = np.concatenate([x1,x2,x3,x4,x5,x6,x7,x8])
        Y = np.concatenate([y1,y2,y3,y4,y5,y6,y7,y8])
        Z = np.concatenate([z1,z2,z3,z4,z5,z6,z7,z8])

        target = np.array([X,Y,Z]).T
        return target


    # get 4 points for each face N/S
    def export_target_8c(self):
        
        n1 = (self.n1 + self.nc) / 2
        n2 = (self.n2 + self.nc) / 2
        n3 = (self.n3 + self.nc) / 2
        n4 = (self.n4 + self.nc) / 2
        s1 = (self.s1 + self.sc) / 2
        s2 = (self.s2 + self.sc) / 2
        s3 = (self.s3 + self.sc) / 2
        s4 = (self.s4 + self.sc) / 2

        tower = np.array([n1,n2,n3,n4,s1,s2,s3,s4])

        target = np.reshape( tower.T,(3, self.N_magnets*8) ).T
        return target



    # 2D target space
    def export_target_n2(self, N, dz=1e-8):
        """
            exports 2 N**2 targets for field sampling
            targets are centered on NxN subdivision of each magnetic face,
            the way there are no edge effects.
            Optional displacement dz=1e-5 available to resolve singularities (default dz=0)
            
            output loops through M magnets, before iterating N**2 face points, then interates other face
            Would doing all samples for each magnet make analysis simpler?
        """
        
        # load data
        L = np.mean(self.L)
        M = self.N_magnets
        H = self.H
    
        # build grid
        ax = ( np.linspace(-1,1,N, endpoint=False) + 1/N ) * (L/2)
        ugrid,vgrid = np.array(np.meshgrid(ax,ax))
        
        # set up local coordinates
        n1 = norm_arr(self.nvec)
        n2 = norm_arr(self.pvec)
        n3 = norm_arr(np.cross(n1,n2))
        r0 = self.com
    
        ux = ugrid[:,:,np.newaxis,np.newaxis]* n2[np.newaxis,np.newaxis,:,:]
        vx = vgrid[:,:,np.newaxis,np.newaxis]* n3[np.newaxis,np.newaxis,:,:]
        uv_grid = np.reshape(ux+vx, (N*N,M,3) )
        
        # transform
        z_height = n1*H[:,np.newaxis]/2 + dz
    
        t_north = r0 + uv_grid + z_height 
        t_south = r0 + uv_grid - z_height 
    
    # (8,M,3)
    #    targets = np.reshape([t_north, t_south], (2*N*N*M,3))
    #    return targets
    
        # shape into (M,8,3)
        targets = np.concatenate([t_north, t_south],axis=0)
        t2 = np.transpose(targets, axes=[1,0,2])
        t3 = np.reshape(t2, (2*N*N*M,3))
        return t3
    
    

    # usage: mlab.triangular_mesh(X,Y,Z, triangle_array, scalars=color_array)
    def export_mayavi_cube(self):

        # define faces for 'C-shape' orientation N=(0,1,2,3) S=(4,5,6,7) 
        cube_form = [(0,1,2),(0,2,3),(0,1,5),(0,4,5),(0,3,7),(0,4,7),(1,2,5),(2,5,6),(2,3,6),(3,6,7),(4,5,6),(4,6,7)]
    
        N = self.N_magnets
        spacer = np.arange(N)*8 
    
        temp = np.array(cube_form)[np.newaxis,:] + spacer[:,np.newaxis,np.newaxis]
        triangle_array = np.concatenate(temp,axis=0)
    
        color = np.array([1,1,1,1,0,0,0,0]) # North red 1 : South blue 0
        color_array = color[np.newaxis,:] * np.ones(N)[:,np.newaxis]
    
        X,Y,Z = np.array([ self.n1,self.n2,self.n3,self.n4, self.s1,self.s2,self.s3,self.s4]).T
        return X,Y,Z, triangle_array, color_array


# this class reads a 3D magnet specified by 16 variables
# x,y,z - r0 center of mass
# nhat  - n1 normal vector parallel to H
# uhat  - n2 normal vector parallel to L
#       - n3 normal vector parallel to W computed from (n1 x n2)
# H,L,W - cuboid dimensions given in (m)
# M     - magnetization strength given in (A m)
# mhat  - magnetization direction
#         * there is some redundancy here, M could be given as a magnitude of mvec = M * mhat
#           I choose this redundancy to make it easier to adjust M independently of orientation.
#           Likewise, H and L could be given as vector magnitudes of n1 and n2.
class Magnet_3D_gen():

    def __init__(self,fname, R=0.3048, HLW=True):

        # read file
        with open(fname) as f:
            datain = f.readlines()
        data  = np.array([ line.strip().split(',') for line in datain[1:] ], float)
        print('Data size:', data.shape)
        self.data = data
        self.N_magnets = len(data)

        # unpack data
        if (HLW):
            x0,y0,z0,nx,ny,nz,ux,uy,uz, H,L,W,M, mx,my,mz = data.T # 16 variable version
        else:
            print('backward compatible, assuming W=L')
            x0,y0,z0,nx,ny,nz,ux,uy,uz, H,L,M, mx,my,mz = data.T
            W = L

        # sort into vectors
        self.r0 = np.array([x0,y0,z0]).T
        n1 = np.array([nx,ny,nz]).T
        n2 = np.array([ux,uy,uz]).T
        m3 = np.array([mx,my,mz]).T

        self.n1 = norm_arr(n1)
        self.n2 = norm_arr(n2)
        self.n3 = norm_arr( np.cross(n1,n2) )
        self.H  =  H
        self.L  =  L
        self.W  =  W
        self.M  =  M
        self.m3 = norm_arr(m3)

    def export_source(self):
        '''
        exports source array for field calculation
        x0,y0,z0,nx,ny,nz,ux,uy,uz, H,L,W,M = source
        '''
        
        # project onto 3 faces
        mvec = self.M[:,np.newaxis] * self.m3
        m1 = np.sum( mvec * self.n1, axis=1 )
        m2 = np.sum( mvec * self.n2, axis=1 )
        m3 = np.sum( mvec * self.n3, axis=1 )
        # also need width
        # H=1, L=2, W=3
        ## this will be a bigger update

        x0,y0,z0 = self.r0.T
        nx,ny,nz = self.n1.T
        ux,uy,uz = self.n2.T
        vx,vy,vz = self.n3.T
        H = self.H
        L = self.L
        W = self.W
        M = self.M

        source = []
        source.append(np.array([x0,y0,z0,nx,ny,nz,ux,uy,uz, H,L,W, m1]).T)
        source.append(np.array([x0,y0,z0,ux,uy,uz,vx,vy,vz, L,W,H, m2]).T)
        source.append(np.array([x0,y0,z0,vx,vy,vz,nx,ny,nz, W,H,L, m3]).T)
        return np.concatenate(source, axis=0)

    # 2D target space
    def export_target_n2(self, N, dz=1e-8):
        """
            exports 2 N**2 targets for field sampling
            targets are centered on NxN subdivision of each magnetic face,
            the way there are no edge effects.
            Optional displacement dz=1e-5 available to resolve singularities (default dz=0)
            
            output loops through M magnets, before iterating N**2 face points, then interates other face
            Would doing all samples for each magnet make analysis simpler?
        """
        
        # load data
        L = np.mean(self.L)
        M = self.N_magnets
        H = self.H
    
        # build grid
        ax = ( np.linspace(-1,1,N, endpoint=False) + 1/N ) * (L/2)
        ugrid,vgrid = np.array(np.meshgrid(ax,ax))
        
        # set up local coordinates
        #n1 = norm_arr(self.nvec)
        #n2 = norm_arr(self.pvec)
        #n3 = norm_arr(np.cross(n1,n2))
        n1 = self.n1
        n2 = self.n2
        n3 = self.n3
        r0 = self.r0
    
        ux = ugrid[:,:,np.newaxis,np.newaxis]* n2[np.newaxis,np.newaxis,:,:]
        vx = vgrid[:,:,np.newaxis,np.newaxis]* n3[np.newaxis,np.newaxis,:,:]
        uv_grid = np.reshape(ux+vx, (N*N,M,3) )
        
        # transform
        z_height = n1*H[:,np.newaxis]/2 + dz
    
        t_north = r0 + uv_grid + z_height 
        t_south = r0 + uv_grid - z_height 
     
        # shape into (M,8,3)
        targets = np.concatenate([t_north, t_south],axis=0)
        t2 = np.transpose(targets, axes=[1,0,2])
        t3 = np.reshape(t2, (2*N*N*M,3))
        return t3

    # export 3M magnets
    def export_m3d(self):
        '''
        exports source array for CIFTJA force calculation
        [x0,y0,z0,nx,ny,nz,ux,uy,uz, H,L,M]
        '''

        x0,y0,z0 = self.r0.T
        nx,ny,nz = self.n1.T
        ux,uy,uz = self.n2.T
        mx,my,mz = self.m3.T
        M = self.M
        H = self.H
        L = self.L
        W = self.W
        
        # x0,y0,z0, nx,ny,nz, ux,uy,uz, H,L,W, M, mx,my,mz = source # new convention
        source = np.array([x0,y0,z0,nx,ny,nz,ux,uy,uz,H,L,W,M, mx,my,mz]).T
        return source

    # writes magnets in vector form
    def write_magnets(self, fout):

        data = self.export_m3d()
        
        print('Preparing to write file')
        with open(fout,'w') as f:
        
            f.write('X [m], Y[m], Z[m], n1x, n1y, n1z, n2x, n2y, n2z, H [m], L [m], W [m], M [A/m], mx, my, mz \n')
            #f.write('X [m], Y[m], Z[m], n1x, n1y, n1z, n2x, n2y, n2z, H [m], L [m], M [A/m], mx, my, mz \n')
            for line in data:
                
                out = '{:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}'.format(*line)
                print(out,file=f)
        print('  Wrote to %s' % fout)
        
    def write_magnets_block(self, fout):
        '''
           converts the 16 vector form (physics) into 24 block form (engineering)
        '''
        R = self.r0
        
        z = self.n1
        x = self.n2
        y = self.n3
        
        h = self.H/2
        l = self.L/2
        w = self.W/2
        
        zh = z*h[:,np.newaxis]
        xl = x*l[:,np.newaxis]
        yw = y*w[:,np.newaxis]
        
        # assume C-shape
        n1 = R + zh + xl + yw
        n2 = R + zh - xl + yw
        n3 = R + zh - xl - yw
        n4 = R + zh + xl - yw
        
        s1 = R - zh + xl + yw
        s2 = R - zh - xl + yw
        s3 = R - zh - xl - yw
        s4 = R - zh + xl - yw
        
        # write
        data = np.array([ *n1.T, *n2.T, *n3.T, *n4.T, *s1.T, *s2.T, *s3.T, *s4.T]).T
        #print(data.shape)
        
        #fout = 'block_'+f_vector
        with open(fout,'w') as f:
        
            head = 'n1x, n1y, n1z, n2x, n2y, n2z, n3x, n3y, n3z, n4x, n4y, n4z, s1x, s1y, s1z, s2x, s2y, s2z, s3x, s3y, s3z, s4x, s4y, s4z'
            print(head, file=f)
            for line in data:
                out = '{:.8f}, {:.8f}, {:.8f}, {:.8f},{:.8f}, {:.8f}, {:.8f}, {:.8f},{:.8f}, {:.8f}, {:.8f}, {:.8f},{:.8f}, {:.8f}, {:.8f}, {:.8f},{:.8f}, {:.8f}, {:.8f}, {:.8f},{:.8f}, {:.8f}, {:.8f}, {:.8f}'.format(*line)
                print(out, file=f)
        print('  Wrote to %s' % fout)

### misc geometry functions
#   copied from MagnetReader.py 8/23

# manipulating magnets
def norm_arr(v):
    v = np.array(v)
    return v / np.linalg.norm(v,axis=-1)[:,np.newaxis]

# does not work for ND arrays
def norm(v):
    v = np.array(v)
    return v / mag(v)

def mag(v):
    return np.sqrt( np.sum(v*v) )
