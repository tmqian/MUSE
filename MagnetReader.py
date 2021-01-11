import numpy as np
import matplotlib.pyplot as plt
from coilpy import *

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
            #self.Lc, self.MT, self.MP] )
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
        print('Wrote %i magents'%N)
        
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
        
        plt.legend()
        
        
    def to_uv(self,R0=0.3048):
        
        x = self.X
        y = self.Y
        z = self.Z
        
        # unpack
        u = np.arctan2(y,x)
        r = np.sqrt( x*x + y*x ) - R0
        v = np.arctan2(z,r) % (np.pi*2)

        return u,v
