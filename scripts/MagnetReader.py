import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc

try:
    from coilpy import *
except:
    print('  note: coilpy package unavailable')

'''
    Last Updated: 27 May 2021
'''

class ReadFAMUS():    

    def __init__(self,fname, N_layers=18):
        self.fname = fname
       
        self.N_layers = N_layers
        self.readfile(fname)
        

    def readfile(self, fname):

        # open
        try:
            with open(fname) as f:
                datain = f.readlines()

        except:
            print('file not found')
            return

                
        # read    
        try:
  
            data = np.array([ line.strip().split(',')[3:12] for line in datain[3:] ], float)
            info = np.array([ line.strip().split(',')[:3] for line in datain[3:] ])
        
        
            # can also try to read N magnets, and q
            if (datain[1].find(',') < 0):
                N_dipoles, q_moment = np.array(datain[1].strip().split(), int)
            else:
                #print('error reading N_mag, q_moment: trying (,) instead of ( )')
                N_dipoles, q_moment = np.array(datain[1].strip().split(','), int)
        
            #ox,  oy,  oz,  Ic,  M_0,  pho,  Lc,  mp,  mt
            X, Y, Z, Ic, M, pho, Lc, MP, MT = np.transpose(data)
            coiltype, symmetry,  coilname = np.transpose(info)
            
            self.type = coiltype
            self.symm = symmetry
            self.name = coilname
            self.X = X
            self.Y = Y
            self.Z = Z
            self.Ic = Ic
            self.Lc = Lc
            self.M = M
            self.pho = pho
            self.MT = MT
            self.MP = MP
            
            self.data = data
            self.info = info
            self.fname = fname[:-6]
            self.N_dipoles = N_dipoles
            self.q    = q_moment
            
            if (N_dipoles != len(data)):
                print('  N_mag != len(data): check famus file')
            
            N_towers = N_dipoles / self.N_layers
            if ((N_towers - int(N_towers)) != 0):
                print('  N_tower - int(N_tower) != 0: check N_layers (default 18)')
            self.N_towers = int(N_towers)

        except:
            print('error reading .focus file')
            return
        
        
    def update_data(self):
            self.data = np.transpose([self.X, self.Y, self.Z,
            self.Ic , self.M,  self.pho,
            self.Lc, self.MP, self.MT] )
            
    def update_info(self):
            self.info = np.transpose([self.type, self.symm, self.name] )
            
    def load_data(self):
        X, Y, Z, Ic, M, pho, Lc, MP, MT = np.transpose(self.data)
        coiltype, symmetry,  coilname = np.transpose(self.info)

        self.type = coiltype
        self.symm = symmetry
        self.name = coilname
        self.X = X
        self.Y = Y
        self.Z = Z
        self.Ic = Ic
        self.Lc = Lc
        self.M = M
        self.pho = pho
        self.MT = MT
        self.MP = MP

    # removes magnets with |rho| = 0 to simplify file
    def skim(self, write=False):

        pho = self.pho
        
        mag = np.array( np.abs(pho) > 1e-4, int )
        #print('Total dipoles:', len(pho))
        #print('Non-zero dipoles:', np.sum(mag))
        
        Ndip = len(pho)
        
        new_data = []
        new_info = []
        
        # there's probably a better way to take subset of array in python
        for j in np.arange(Ndip):
            if (mag[j] == 0):
                continue
            new_data.append( self.data[j] )
            new_info.append( self.info[j] )
        
        self.data = np.array(new_data)
        self.info = np.array(new_info)
        self.load_data()
        
        if (write): 
            fout = self.fname + '_skim.focus' 
            self.writefile(fout)
   
    # unfinished
    def halfperiod_to_fulltorus(self):
        x = self.X 
        y = self.Y 
        z = self.Z 
        p = self.pho 
        X,Y,Z,P = stellarator_symmetry(x,y,z,p)

        ## inside stellarator_symmetry
        #X = np.concatenate((x,-x,-x,x))
        #Y = np.concatenate((y,y,-y,-y))
        #Z = np.concatenate((z,-z,z,-z))
        #M = np.concatenate((m,-m,m,-m))


    # does not modify pho, but writes whatever is there
    def writefile(self, fname, q=1):
        
        N = len(self.pho)
        h1 = '# Total number of coils,  momentq \n'
        h2 = '     {}     {}\n'.format(N,q) # removed comma to match FAMUS
        h3 = '# coiltype, symmetry,  coilname,  ox,  oy,  oz,  Ic,  M_0,  pho,  Lc,  mp,  mt \n'
        
        outdata = np.transpose([self.type, self.symm, self.name,
                               self.X, self.Y, self.Z,
                               self.Ic, self.M, self.pho,
                               self.Lc, self.MP, self.MT])
        with open(fname,'w') as f:
            f.write(h1)
            f.write(h2)
            f.write(h3)
            
            for j in np.arange(N):
                line = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} \n'.format(*outdata[j])
                f.write(line)
        print('Wrote %i magnets'%N)
        print('  new file:', fname)
        
        
    def hist_pho(self,q=1,bins=20,new_fig=False):
        
        if (new_fig):
            plt.figure()
        a_rho = abs(self.pho**q)
        
        
        chid = np.mean( ( a_rho*(1 - a_rho) )**2 * 4 )
        plt.hist(a_rho, bins, label='dpbin = %.2e'%chid)
        
        
        # calculate pmvol
        m = self.M[0]
        s = np.sum(np.abs(self.pho)) 
        pmvol = s*m*4 # full torus
        plt.plot([],[],'C1*',label='pmvol = %.3e' % pmvol)
        
        fn = self.fname.split('/')[-1].split('.')[0]
        plt.title(fn)
        
        plt.legend()
        
        
    def to_uv(self,R0=0.3048):
        # converts 3D magnet position to 2D solid angle
        # u is toroidal angle, v is poloidal angle.
        
        x = self.X
        y = self.Y
        z = self.Z
        
        # unpack
        u = np.arctan2(y,x)
        r = np.sqrt( x*x + y*y ) - R0
        v = np.arctan2(z,r) % (np.pi*2)

        return u,v
    
    def to_towers(self,N_layers=18,q=1,R0=0.3048):
    
        N = len(self.pho)

        N_towers = int(N/N_layers)
        N_surf = int(N / N_layers)


        # get data
        rho = np.reshape( self.pho**q, (N_layers,N_surf) )
        xx = np.reshape( self.X, (N_layers,N_surf) )
        yx = np.reshape( self.Y, (N_layers,N_surf) )
        zx = np.reshape( self.Z, (N_layers,N_surf) )

        # unpack
        px = np.arctan2(yx,xx)
        rx = np.sqrt( xx*xx + yx*yx ) - R0
        tx = np.arctan2(zx,rx)

        # shift
        tx2 = tx % (np.pi*2)

        pho = np.array(self.pho > 0.9,int) - np.array(self.pho < -0.9,int)
        layers = np.reshape(pho,(N_layers,N_towers))
        projec = np.sum(layers,axis=0)

        return px[0],tx2[0],projec

 
    def slice_map(self,N_layers=18):
        # Plots magnet distribution, layer by layer
        u,v,projec = self.to_towers(N_layers=N_layers)
        pho = self.pho
        N_towers = int(len(pho)/ N_layers)
        print(len(pho), N_towers)
        m = np.reshape(self.pho, (N_layers,N_towers))

        plt.figure(figsize=(9,12))

        plt.subplot(5,4,1)
        plt.tricontourf(u,v,projec,N_layers,cmap='RdBu_r',extend='both')
        plt.colorbar()
        plt.title('Total Towers')
        plt.xlabel('toroidal half period')
        plt.ylabel('poloidal angle')

        plt.axhline(np.pi/2,ls='--',color='C1')
        plt.axhline(3*np.pi/2,ls='--',color='C1')
        plt.axvline(np.pi/4,ls='--',color='C1')

        levels = np.linspace(-.9,.9,10)
        #levels = [-.9,-.7,-.5,-.3,-.1,.1,.3,.5,.7,.9]
        norm = mc.BoundaryNorm(levels, 256)
 
        for s in np.arange(N_layers):
            plt.subplot(5,4,s+2)
            plt.tricontourf(u,v,m[s],cmap='jet',extend='both', levels=levels, norm=norm) 
            #plt.clim(-1,1) 
            plt.colorbar()
            plt.title('Layer %i' % (s+1) )
         #   plt.xlabel('toroidal half period')
         #   plt.ylabel('poloidal angle')

            plt.axhline(np.pi/2,ls='--',color='C1',lw=0.7)
            plt.axhline(3*np.pi/2,ls='--',color='C1',lw=0.7)
            plt.axvline(np.pi/4,ls='--',color='C1',lw=0.7)


        plt.suptitle(self.fname)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
 
    def dpbin_map(self,N_layers=18):
        # Plots magnet distribution, layer by layer
        u,v,projec = self.to_towers(N_layers=N_layers)
        pho = self.pho
        N_towers = int(len(pho)/ N_layers)
        #print(len(pho), N_towers)

        plt.figure(figsize=(9,12))

        plt.subplot(5,4,1)
        plt.tricontourf(u,v,projec,N_layers,cmap='RdBu_r',extend='both')
        plt.colorbar()
        plt.title('Total Towers')
        plt.xlabel('toroidal half period')
        plt.ylabel('poloidal angle')

        plt.axhline(np.pi/2,ls='--',color='C1')
        plt.axhline(3*np.pi/2,ls='--',color='C1')
        plt.axvline(np.pi/4,ls='--',color='C1')

        levels = np.linspace(-.9,.9,10)
        #levels = [-.9,-.7,-.5,-.3,-.1,.1,.3,.5,.7,.9]
        norm = mc.BoundaryNorm(levels, 256)

        m = np.reshape(self.pho, (N_layers,N_towers))

        sgn = np.array(m>0,int)*2 - 1
        M = np.abs(m)
        dpbin = M*(1-M) * sgn

        for s in np.arange(N_layers):
            plt.subplot(5,4,s+2)
            plt.tricontourf(u,v,10*dpbin[s],cmap='jet',extend='both', levels=levels, norm=norm) 
            #plt.tricontourf(u,v,dpbin[s],cmap='jet',extend='both')#, levels=levels, norm=norm) 
            #plt.clim(-1,1) 
            plt.colorbar()
            plt.title('Layer %i' % (s+1) )
         #   plt.xlabel('toroidal half period')
         #   plt.ylabel('poloidal angle')

            plt.axhline(np.pi/2,ls='--',color='C1',lw=0.7)
            plt.axhline(3*np.pi/2,ls='--',color='C1',lw=0.7)
            plt.axvline(np.pi/4,ls='--',color='C1',lw=0.7)


        plt.suptitle('10x dpbin: '+self.fname)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        
    def plot_symm(self, export=False, fig=True, show_pipes=True,N_layers=18):

        tx,px,m = self.to_towers(N_layers=N_layers)
        

        T = np.concatenate((tx, np.pi-tx[::-1])) # toroidal
        P = np.concatenate((px,px))              # poloidal
        M = np.concatenate((m,-m[::-1]))

        T2 = np.concatenate((T,np.pi+T))
        P2 = np.concatenate((P,P))
        M2 = np.concatenate((M,M))
        
        if (export):
            return T2,P2,M2
        
        if (fig):
            plt.figure()
 
        #levels = np.linspace(-15,15,8)
        #levels = np.linspace(-N_layers, N_layers, N_layers+1)
        levels = [-15,-11,-7,-5,-1,1,5,7,11,15]
        norm = mc.BoundaryNorm(levels, 256)
        plt.tricontourf(T2,P2,M2,cmap='jet',extend='both', levels=levels, norm=norm) 
 
#        plt.tricontourf(T2,P2,M2,10,extend='both',cmap='RdBu_r')
        plt.colorbar()
              
        if (show_pipes):
            plt.axvline(np.pi/4,ls='--',color='C1')
            plt.axvline(5*np.pi/4,ls='--',color='C1')

            plt.axvline(3*np.pi/4,ls='--',color='C1')
            plt.axvline(7*np.pi/4,ls='--',color='C1')
    
                
                
    def load_rings(self, N_poloidal=89):

        u,v = self.to_uv()
        v_idx = np.array(v*100,int)

        rings = []
        j = 0
        while j < self.N_towers:

            count = 1
            while ( np.abs(v[j+1] - v[j]) < 1./N_poloidal):
                count += 1
                j += 1

            rings.append(count)
            j += 1

        self.rings = np.array(rings)
        return np.array(rings)
              
                    
    def load_dots(self):
        
        rings = self.load_rings()
    
        dots = []
        y = 0
        for k in rings:
            for x in np.arange(k):
                dots.append((x,y))
            y += 1
            
        return np.array(dots)
    

    '''
        u: j toroidal
        v: i poloidal
        w: k slice

        note - (u,v) starts from 0 while w starts from 1
        such that w=5 points to the 5th layer
    '''
    def idx_3d(self,u,v,w):
        
        try:
            rings = self.rings
        except:
            print('  initializing poloidal rings')
            rings = self.load_rings()
        
        return u + np.sum(rings[:v]) + self.N_towers*(w-1)

### end class function
def mayavi_plot_rho(self,scale=0.00635, show_vector=False, vec_scale=0.05, 
                    add_symmetry=False, flip_sign=False, skip_switch=False, filter_blank=0,
                    q=1, legend=True, plot_M=False):
    
    from mayavi import mlab
    
    if (skip_switch):
        self.skim()

    X, Y, Z, Ic, M, pho, Lc, MP, MT = np.transpose(self.data)
    
#     if (skip_switch):
#         X = X * Ic
#         Y = Y * Ic
#         Z = Z * Ic
#         pho = pho * Ic
        
    if (filter_blank > 0):
        
        isMag = np.array( abs(pho) > filter_blank, int )
        
        X = X * isMag
        Y = Y * isMag
        Z = Z * isMag
        #pho = pho *

    if (add_symmetry):
        X,Y,Z,pho = stellarator_symmetry(X,Y,Z,pho)

    rho = pho**q
    if (plot_M):
        rho = rho*M
    mlab.points3d(X,Y,Z,rho,scale_mode='none',scale_factor=scale)

    if (show_vector):

        X, Y, Z, Ic, M, pho, Lc, MP, MT = np.transpose(self.data)

        U = np.sin(MT) * np.cos(MP)
        V = np.sin(MT) * np.sin(MP)
        W = np.cos(MT)

        if (flip_sign):
            U,V,W = flip_magnets(U,V,W,pho)
        mlab.quiver3d(X,Y,Z,U,V,W,color=(1,1,0), scale_factor=vec_scale)

    if(legend):
        mlab.scalarbar()

    mlab.show()

    
# applies stellarator symmetry and plots VV
# untested
def mayavi_full_torus(self):
    
    self.skim()
    x,y,z,p = stellarator_symmetry(self.ox,self.oy,self.oz,self.pho)
    
    def make_torus():
        # I want to plot a solid torus, representing the VV, for the filter view
        N_zeta = 256
        N_theta = 64
        U = np.linspace(0,np.pi*2,N_zeta)#, endpoint=False)
        V = np.linspace(0,np.pi*2,N_theta)#, endpoint=False)
        a = 0.0847
        r = 0.3048

        R = r + a*np.cos(V)
        H = a*np.sin(V)

        X = [[ rr*np.cos(u) for u in U ] for rr in R]
        Y = [[ rr*np.sin(u) for u in U ] for rr in R]
        Z = [[ h            for u in U ] for h  in H]
        
        return X,Y,Z
    
    X,Y,X = make_torus()
    mlab.mesh(X,Y,Z, color=(1,1,1))
   
    mlab.points3d(x,y,z,p,scale_mode='none',scale_factor=0.005)
    mlab.show()


    
'''
   takes a half period, creates full torus (assuming NFP=2)
   may also assume which half period we start with
'''
def stellarator_symmetry(x, y, z, m):

    X = np.concatenate((x,-x,-x,x))
    Y = np.concatenate((y,y,-y,-y))
    Z = np.concatenate((z,-z,z,-z))
    M = np.concatenate((m,-m,m,-m))
    
    return X, Y, Z, M

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

# performs quarternion rotation of v, around direction n, by angle t (positive right-hand rotation)
def rotate(v,n,t):
    n = norm(n)
    r = v*np.cos(t) + np.cross(n,v)*np.sin(t) + (1-np.cos(t))*np.dot(v,n)*n
    return r


def write_scad(data, fname='test.txt'):
    # format: data = np.transpose((X,Y,Z,u,v,w,m))
    
    with open(fname,'w') as f:
        f.write('N_DIP = %i; \n'% len(data) )
        f.write('data = [')
        for line in data[:-1]:
            if line[-1]==0:
                continue
            f.write('[ {}, {}, {}, {:.6}, {:.6}, {:.6}, {}],\n'.format(*line))

        f.write('[ {}, {}, {}, {:.6}, {:.6}, {:.6}, {}]];'.format(*(data[-1]) ))


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


### new file format
'''
    This reads Doug's file format, where magnets are saved as 3D volumes.
    Each magnet is encoded by 8 points in space, representing the corners.
    The convention is n1,n2,n3,n4 followed by s1,s2,s3,s4, where the difference between faces
    is used to encode orientation. This assumes the PM is magnetized parallel to axis.
'''
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

        color = np.array([0,0,0,0,1,1,1,1]) # North red 0 : South blue 1
        color_array = color[np.newaxis,:] * np.ones(N)[:,np.newaxis]

        X,Y,Z = np.array([ self.n1,self.n2,self.n3,self.n4, self.s1,self.s2,self.s3,self.s4]).T
        return X,Y,Z, triangle_array, color_array


class Magnet_3D_gen():

    def __init__(self,fname, R=0.3048):

        # read file
        with open(fname) as f:
            datain = f.readlines()
        data  = np.array([ line.strip().split(',') for line in datain[1:] ], float)
        print('Data size:', data.shape)
        self.data = data
        self.N_magnets = len(data)

        # unpack data
        x0,y0,z0,nx,ny,nz,ux,uy,uz, H,L,M, mx,my,mz = data.T

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
        self.W  =  L   # assumes input is square plate (temporary)
        self.M  =  M
        self.m3 = norm_arr(m3)

    # for the Ciftja force (backward compatible)
    def export_source(self):
        '''
        exports source array for CIFTJA force calculation
        [x0,y0,z0,nx,ny,nz,ux,uy,uz, H,L,M]
        '''

        mvec = self.M[:,np.newaxis] * self.m3
        m1 = np.sum( mvec * self.n1, axis=1 )
        m2 = np.sum( mvec * self.n2, axis=1 )
        m3 = np.sum( mvec * self.n3, axis=1 )
        # also need width
        # H=1, L=2, W=3A
        ## this will be a bigger update

        x0,y0,z0 = self.r0.T
        nx,ny,nz = self.n1.T
        ux,uy,uz = self.n2.T
        H = self.H
        L = self.L
        M = self.M

        source = np.array([x0,y0,z0,nx,ny,nz,ux,uy,uz, H,L,M]).T
        return source

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

    def write_magnets(self, fout):

        data = self.export_m3d()
        
        print('Preparing to write file')
        with open(fout,'w') as f:
        
            f.write('X [m], Y[m], Z[m], n1x, n1y, n1z, n2x, n2y, n2z, H [m], L [m], M [A/m], mx, my, mz \n')
            for line in data:
                
                out = '{:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}'.format(*line)
                print(out,file=f)
        print('  Wrote to %s' % fout)
        
