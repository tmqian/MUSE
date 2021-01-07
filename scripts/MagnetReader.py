import numpy as np
import matplotlib.pyplot as plt

try:
    from coilpy import *
except:
    print('  note: coilpy package unavailable')
#from mayavi import mlab

'''
    Last Updated: 03 Jan 2021
'''

class ReadFAMUS():    

    def __init__(self,fname):
        self.fname = fname
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

        except:
            print('error reading .focus file')
            return
        
        
    def update_data(self):
            self.data = np.transpose([self.X, self.Y, self.Z,
            self.Ic , self.M,  self.pho,
            self.Lc, self.MP, self.MT] )
            
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
      
    # does not modify pho, but writes what ever is there
    def writefile(self, fname, q=1):
        
        N = len(self.pho)
        h1 = '# Total number of coils,  momentq \n'
        h2 = '     {},     {}\n'.format(N,q)
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
        
    def mayavi_plot_rho(self,scale=0.03, show_vector=True, vec_scale=0.05, \
                        add_symmetry=False, flip_sign=False, q=1, legend=True):
        
        X, Y, Z, Ic, M, pho, Lc, MP, MT = np.transpose(self.data)
        
        if (add_symmetry):
            X,Y,Z,pho = stellarator_symmetry(X,Y,Z,pho)
        
        rho = pho**q
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


    def plot_slices(self,N_layers=18):

        u,v,projec = self.to_towers(N_layers=N_layers)

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


        for s in np.arange(1,20):
            plt.subplot(5,4,s+1)

            plt.tricontourf(u,v,
                            np.array(abs(projec)< s+.1, int)
                            *np.array(abs(projec)>s-.1, int),N_layers,cmap='plasma')#,extend='both')
          #  plt.colorbar()
            plt.title('exactly %i slice'%s)
         #   plt.xlabel('toroidal half period')
         #   plt.ylabel('poloidal angle')

            plt.axhline(np.pi/2,ls='--',color='C1',lw=0.7)
            plt.axhline(3*np.pi/2,ls='--',color='C1',lw=0.7)
            plt.axvline(np.pi/4,ls='--',color='C1',lw=0.7)


        #plt.suptitle(fd.fname)
        plt.tight_layout()
        
        
        
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

        plt.tricontourf(T2,P2,M2,10,extend='both',cmap='RdBu_r')
        plt.colorbar()
        
        if (show_pipes):
            plt.axvline(np.pi/4,ls='--',color='C1')
            plt.axvline(5*np.pi/4,ls='--',color='C1')

            plt.axvline(3*np.pi/4,ls='--',color='C1')
            plt.axvline(7*np.pi/4,ls='--',color='C1')
        

### end class function
def mayavi_plot_rho(self,scale=0.00635, show_vector=False, vec_scale=0.05, \
                    add_symmetry=False, flip_sign=False, q=1, legend=True):
    
    from mayavi import mlab

    X, Y, Z, Ic, M, pho, Lc, MP, MT = np.transpose(self.data)

    if (add_symmetry):
        X,Y,Z,pho = stellarator_symmetry(X,Y,Z,pho)

    rho = pho**q
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
