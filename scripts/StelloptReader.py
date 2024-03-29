import numpy as np
import matplotlib.pyplot as plt

from netCDF4 import Dataset
#from coilpy import FourSurf
try:
    from surface import FourSurf
except:
    print('no surface reader library')
# Updated 26 June 2021

### Define Angles
#      User input, ADJUST these!
N_toroidal_Booz = 256 
N_poloidal_Booz = 64

N_poloidal_VMEC = 128

# choose plasma boundary
#fboundary = '../POLYHYMNIA/ncsx_2p_vacuum.plasma'
#plasma = FourSurf.read_focus_input(fboundary) # I don't like this, write your own function


# function for reading netCDF output
def get(f,key):
    return f.variables[key][:]


# Taking Fourier modes CMN, selects for flux surface s_idx
# select sine or cosine for array. 
#      This global function is used to fourier transform 
#      either Rmnc or Zmns in the VMEC class,
#      or Bmnc in the BOOZ class.
# input toroidal and poloidal angle axis (tax, pax)
# outputs 2D array Z(p,t)
def fourier2space(Cmn, tax,pax, xm,xn, s_idx=48, N_modes=61, sine=True):
    
    arr = []
    
    #global xm,xn
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

def plot_circle(R=0.3048,a=0.0762,N=100):
    tx = np.linspace(0,np.pi*2,N)
    
    r = [R + a*np.cos(t) for t in tx]
    z = [a*np.sin(t) for t in tx]
    
    plt.plot(r,z,'k--')

'''
NEO
'''

class readNEO():
    
    def __init__(self,fname):

        # read
        with open(fname) as f:
            datain = f.readlines()

        data = np.array([line.strip().split() for line in datain],float)
        surf, eps_eff, reff, iot, b_ref, r_ref = np.transpose(data)

        self.surf = surf
        self.eps_eff = eps_eff # actually this data is eps_eff^3/2

        self.ns = len(surf)


# old for legacy scripts
# maybe I should upgrade this into a class
def read_neo(f):
    
    with open(f) as h:
        datain = h.readlines()

    data = np.array([line.strip().split() for line in datain],float)
    surf, eps_eff, reff, iot, b_ref, r_ref = np.transpose(data)
    return surf,eps_eff

# plots eps_eff**3/2 against flux radius
def plot_neo(f,ref=False):
    #f = files_n[i]
    surf,eps_eff = read_neo(f)
    
    N = len(surf)
    N = 48
    #sx = np.linspace(0,1,N)
    
    s = np.array([16,26,40])
    
    if (ref):
        #e = np.array([7.402E-04, 8.604E-04, 1.054E-03]) # CFQS
        e = np.array([5.64E-04, 7.86E-04, 1.20E-03]) # NCSX
        plt.plot(s/N,e,'C8*',label='fixed boundary target NCSX')
    
    tag = f.split('/')[-1][8:]
    plt.plot(surf/N,eps_eff,'.--',label=tag)
    plt.yscale('log')
    
    plt.xlabel('flux surface s',fontsize=14)
    plt.ylabel(r'$\varepsilon_{eff}^{3/2}$',fontsize=14)
    plt.title('Neoclassical Transport',fontsize=14)

    
    plt.grid(True)
    #plt.title('EPS_EFF')
    plt.legend()
    plt.xlim(0,1)
    plt.tight_layout()

# plots eps_eff against radius
def plot_neo_rad(f,ref=False):
    #f = files_n[i]
    surf,eps_eff = read_neo(f)
    
    N = len(surf)
    N = 48
    #sx = np.linspace(0,1,N)
    
    s = np.array([16,26,40])
    
    if (ref):
        #e = np.array([7.402E-04, 8.604E-04, 1.054E-03]) # CFQS
        e = np.array([5.64E-04, 7.86E-04, 1.20E-03]) # NCSX
        plt.plot( np.sqrt(s/N), e**(2/3.) ,'C8*',label='fixed boundary target NCSX')
        #plt.plot( s/N,e,'C8*',label='fixed boundary target NCSX')
    
    tag = f.split('/')[-1][8:]
    plt.plot( np.sqrt(surf/N), eps_eff**(2/3.), '.--',label=tag)
    #plt.plot(surf/N,eps_eff,'.--',label=tag)
    plt.yscale('log')
    
    plt.xlabel('normalized radius (r/a)',fontsize=14)
    plt.ylabel(r'$\varepsilon_{eff}$',fontsize=14)
    plt.title('Neoclassical Transport',fontsize=14)

    
    plt.grid(True)
    #plt.title('EPS_EFF')
    plt.legend()
    plt.xlim(0,1)
    plt.tight_layout()
    
    
'''
BOOZ
'''

class readBOOZ():
    
    def __init__(self,fname, Nt=N_toroidal_Booz, Np=N_poloidal_Booz):
        self.fname = fname
        self.f = Dataset(fname, mode='r')
        self.tag = fname.split('/')[-1][7:-3]
        
        self.Nt = Nt
        self.Np = Np
        
        f = self.f
        self.ns    = get(f,'ns_b')
        self.slist = get(f,'jlist')
        self.xm    = get(f,'ixm_b')
        self.xn    = get(f,'ixn_b')
        self.bmnc  = get(f,'bmnc_b')
        self.iota_b  = get(f,'iota_b')

        self.S = self.calc_S()
        

#    def calc_S(self):
#      
#        # get asymetric modes (assumes axisymmetry!)
#        arg_asym = np.argwhere(self.xn != 0)[:,0]
#        
#        # get background mode for normalization
#        b00 = self.bmnc[:,0]  # does this also assume the backgroud mode is axisymetric?
#        bmn_norm = self.bmnc / b00[:,np.newaxis]
#
#        # comput rms
#        #S = np.sqrt(np.sum(bmn_norm[:,arg_asym]**2, axis=1))
#        S = np.linalg.norm(bmn_norm[:,arg_asym], axis=1)
#        return S

    # calculate Landreman Martin metric for quasisymmetry on each surface
    def calc_S(self, N=0,M=1):
      
        # get asymetric modes 
        #symm     =  np.array(M*self.xn != N*self.xm,int)
        arg_asym = np.argwhere( M*self.xn != N*self.xm )[:,0]
        
        # get background mode for normalization
        b00 = self.bmnc[:,0]  # b00 background for EACH surface
        bmn_norm = self.bmnc / b00[:,np.newaxis]
        #bmn_norm = self.bmnc / np.max(self.bmnc)

        # comput rms
        S = np.linalg.norm(bmn_norm[:,arg_asym], axis=1)
        return S
         
    def plot_Booz_Contour(self, s_idx=1, fig=True, plot_iota=False, cmap='plasma',tag_fname=True):
        
        # grid
        tax = np.linspace(0, 2*np.pi, self.Nt, endpoint=False)
        pax = np.linspace(0, 2*np.pi, self.Np, endpoint=False)

        # calc
        N = len(self.bmnc)
        modB = fourier2space(self.bmnc, tax, pax, self.xm, self.xn, sine=False, s_idx=s_idx) 

        # plot
        if (fig):
            plt.figure()
        plt.contourf(tax,pax,modB,cmap=cmap)
        plt.xlabel(r'Boozer Toroidal $\zeta$',fontsize=14)
        plt.ylabel(r'Boozer Poloidal $\theta$',fontsize=14)
        
        s = self.slist[s_idx] / self.ns
        plt.title(r'$|B|$ [T] (s = %.2f)'% s ,fontsize=14)
        plt.colorbar()#(label='|B| [T]')
        #plt.clim(.12,.16)

        if (tag_fname):
            plt.plot([],[],'w.',label=self.tag)

        if (plot_iota):
            iota = self.iota_b[ self.slist[s_idx] ]

            iota_trace = (iota*tax + np.pi) % (2*np.pi)
            #plt.plot(tax, iota_trace, 'C2.', label = 'iota = %.3f'%iota)
            plt.plot(tax, iota_trace, 'C2--', label = 'iota = %.3f'%iota)
            #plt.plot(tax, iota*tax + np.pi, 'C2--', label = 'iota = %.3f'%iota)

        plt.legend()
       
    # This plots ripple well,
    #     not to be confused with magnetic well from MHD stability   
    def plot_B_well(self, s_idx=1, nfp=20, fig=True):
        
        tax_b = np.linspace(0,nfp*np.pi,1000)
        iota  = self.iota_b[ self.slist[s_idx] ]
        s = self.slist[s_idx] / self.ns

        def quick_p(t,iota):
            return t*iota + np.pi

        # compute well
        modB = []
        n_modes = len(self.xm)
        for j in np.arange(n_modes):

            m = int( self.xm[j] )
            n = int( self.xn[j] )

            b_amplitude = self.bmnc[s_idx,j]
            b = [ b_amplitude * np.cos( m*quick_p(t,iota) - n*t )  for t in tax_b] # is my sgn convention correct?

            modB.append(b)
        modB = np.sum(modB, axis=0)
        
        # plot well
        if (fig):
            plt.figure()
        plt.plot(tax_b/np.pi, modB, label=self.tag)
        plt.title('|B| along field line (s=%.2f)' % s, fontsize=14)
        #plt.title('Magnetic Well (s=%.2f)' % s, fontsize=14)
        plt.xlabel('N field periods', fontsize=14)
        plt.ylabel('|B| [T]', fontsize=14)
        plt.legend(loc=2)
        plt.tight_layout()
        
        
    def plot_booz_spectrum(self, n_modes=10, fig=True,s_idx=1):
  
        sax = self.slist / self.ns
        bmn = self.bmnc[s_idx]
        big_idx = np.argsort( abs(bmn) ) [::-1] [1:n_modes+1]
        
    
        if (fig):
            plt.figure(figsize=(8,4))
        
        plt.plot([],[],'w.',label=self.tag)

        bmax = 0.15
        #bmax = np.max(np.abs(self.bmnc[:, big_idx[0] ])) # normalize to largest mode
        for j in big_idx:
            m = self.xm[j]
            n = self.xn[j]
            b_mode = bmn[j]
            plt.plot(sax, self.bmnc[:,j]/bmax,'.-',label='(%i,%i)'%(n,m))

        s = self.slist[s_idx] / self.ns
        plt.axvline(s, ls='--',color='k',lw=0.7, label='sorted by s=%.2f' % s )

        plt.legend(loc=2,frameon=False)

        plt.xlabel(r'flux surface $s$', fontsize=14)
        plt.ylabel(r'$b_{nm} / B_0$', fontsize=14)
        #plt.ylabel(r'$b_{nm} / b_{max}$', fontsize=14)
        #plt.ylabel(r'$b_{nm}$', fontsize=14)
        plt.title('Field amplitude (n,m)', fontsize=14)
        
        plt.xlim(-.05,1.05)
        plt.tight_layout()
        
    def plot_booz_spectrum_log(self, n_modes=10, fig=True,s_idx=1):
  
        sax = self.slist / self.ns
        bmn = self.bmnc[s_idx]
        big_idx = np.argsort( abs(bmn) ) [::-1] [1:n_modes+1]
        
    
        if (fig):
            plt.figure(figsize=(8,4))
        
        plt.plot([],[],'w.',label=self.tag)

        bmax = 0.15
        #bmax = np.max(np.abs(self.bmnc[:, big_idx[0] ])) # normalize to largest mode
        for j in big_idx:
            m = self.xm[j]
            n = self.xn[j]
            b_mode = bmn[j]
            plt.plot(sax, np.abs(self.bmnc[:,j]/bmax),'.-',label='(%i,%i)'%(n,m))

        s = self.slist[s_idx] / self.ns
        plt.axvline(s, ls='--',color='k',lw=0.7, label='sorted by s=%.2f' % s )

        plt.legend(loc=2,frameon=False)
        plt.yscale('log')

        plt.xlabel(r'flux surface $s$', fontsize=14)
        plt.ylabel(r'$log( b_{nm} / B_0 )$', fontsize=14)
        #plt.ylabel(r'$b_{nm} / b_{max}$', fontsize=14)
        #plt.ylabel(r'$b_{nm}$', fontsize=14)
        plt.title('Field amplitude (n,m)', fontsize=14)
        
        plt.xlim(-.05,1.05)
        plt.tight_layout()
        
#     def plot_1d_hist(self):
        
#         if (fig):
#             plt.figure()

#         arr = np.ndarray.flatten(self.bmnc)
#         plt.hist( np.log(abs(arr)), 100)

#         plt.title('total modes: %i' % len(self.xm))
#         plt.xlabel('log |B|')
        
    def plot_2d_modes(self, s_idx=1, fig=True):

        if (fig):
            plt.figure()

        plt.tricontourf(self.xn, self.xm, np.log(abs(self.bmnc[s_idx])), cmap='jet')
        plt.colorbar()
        
        plt.plot([],[],'w.',label=self.tag)
        plt.legend(loc=2)

        plt.xlabel(r'Boozer Toroidal $n$',fontsize=14)
        plt.ylabel(r'Boozer Poloidal $m$',fontsize=14)

        s = self.slist[s_idx] / self.ns
        plt.title(r'$\log |B|$  (s = %.2f)'% s ,fontsize=14)
    
'''
VMEC
'''
    
class readVMEC():
    def __init__(self,fname,R=0.3048,a=0.0762):

        self.fname = fname
        self.f = Dataset(fname, mode='r')
        self.tag = fname.split('/')[-1][5:-3]
        
        self.R = R
        self.a = a
        
        f = self.f
        self.ns = get(f,'ns')
        self.xm = get(f,'xm')
        self.xn = get(f,'xn')
        self.rmnc = get(f,'rmnc')
        self.zmns = get(f,'zmns')
        self.iota = get(f,'iotaf')
        self.pressure = get(f,'presf')
        self.nfp = get(f,'nfp')

        self.Aminor = get(f,'Aminor_p')
        self.Rmajor = get(f,'Rmajor_p')
        #self.N_modes = len(self.xm)
        self.N_modes = int(self.f.variables['mnmax'][:])

        self.plasma_file = False
        
    def get_surface(self, N, phi=0, s=-1):
        
        pax = np.linspace(0,np.pi*2,N)
        tax = np.array([phi])

        N_modes = self.N_modes

        # positions
        R2d = fourier2space(self.rmnc, tax,pax, self.xm, self.xn, sine=False, s_idx=s, N_modes=N_modes)
        Z2d = fourier2space(self.zmns, tax,pax, self.xm, self.xn, sine=True,  s_idx=s, N_modes=N_modes)

        # cartisian coordinates for flux surface
        R = R2d [:,0]
        Z = Z2d [:,0]

        return R,Z
        
    # Consider the asymmetric case, where there is for example rmnc AND rmns
    # input toroidal and poloidal angle axis (tax, pax)
    # outputs 2D array R(p,t)
    #     this can be accelerated and reduced to 1D for a tokamak
    def fourier2space_asym(self, cos_mn, sin_mn, tax,pax, s_idx=48):

        arr = []
        
        xm = self.xm
        xn = self.xn
        for j in np.arange(self.N_modes):
    
            m = int( xm[j] )
            n = int( xn[j] )
            
            c = cos_mn[s_idx,j]
            s = sin_mn[s_idx,j]
    
            A = np.array( [[ s * np.sin( m*p - n*t )  for t in tax] for p in pax ] )
            B = np.array( [[ c * np.cos( m*p - n*t )  for t in tax] for p in pax ] )
            
            arr.append(A + B)
    
        return np.sum(arr, axis=0)

    def get_surface_asym(self, N, phi=0, s=-1):
        
        pax = np.linspace(0,np.pi*2,N)
        tax = np.array([phi])

        rmns = get(self.f, 'rmns')
        zmnc = get(self.f, 'zmnc')

        # positions
        R2d = self.fourier2space_asym(self.rmnc, rmns, tax,pax, s_idx=s)
        Z2d = self.fourier2space_asym(zmnc, self.zmns, tax,pax, s_idx=s)

        # cartisian coordinates for flux surface
        R = R2d [:,0]
        Z = Z2d [:,0]

        return R,Z

    def plot_single_angle(self,phi,color='C0',legend=True):
        for s in [1,5,10,15,20,25,30,35,40,45,-1]:
            R,Z = self.get_surface(N_poloidal_VMEC ,s=s, phi=phi)
            #plt.plot(R*100,Z*100,'%s--'%color,lw=0.7)
            plt.plot(R,Z,'%s--'%color,lw=0.7)


        #plot_circle(R=100*self.R,  a=100*self.a)
        plt.axis('equal')
        plt.title(r'$\zeta = %.2f$'%phi)
        plt.axis('square')
        plt.ylabel('Z [m]', fontsize=12)
        plt.xlabel('R [m]', fontsize=12)
        
        if (self.plasma_file):
            self.plasma_plot(zeta=phi,npoints=360)
        
        if (legend):
            plt.plot([],[],'%s--'%color, label=self.tag)
            plt.legend(loc=2,frameon=False)
        else: 
            plt.plot([],[],'%s--'%color)
        

    def plot_vmec_3(self):
    #def plot_vmec_3(self, phi=[0,np.pi/4, np.pi/2]):
    
        plt.figure(figsize=(11,3.5))

        phi = np.array([0,0.5,1]) * np.pi / self.nfp  
        N = len(phi)
        colors = ['C0','C1','C2']

        legend = True
        for k in np.arange(N):
            plt.subplot(1,N,k+1)
            self.plot_single_angle(phi[k], color=colors[k], legend=legend)
            legend = False
        plt.tight_layout()
        
    def plot_iota(self, fig=True,ref=True):
    
        if (fig):
            plt.figure()
    

        if (ref):
            # fixed ref for NCSX
            #plt.plot([0,1],[0.199,0.188],'C8*--',label='target ncsx')
            plt.plot([0,1],[0.182,0.197],'C8*--',label='target pg19')
            #plt.plot([0,48],[0.199,0.188],'C8*--',label='target ncsx')
            
        N = len(self.iota)
        s = np.arange(N) + 1
        plt.plot(s/N, self.iota, '.--',label=self.tag)
        plt.legend(frameon=False)

        plt.title('Rotation Transform')
        plt.ylabel('iota')
        plt.xlabel('surface')
        plt.grid()

    def load_plasma(self, plasma_file):

        self.plasma_file = plasma_file
        self.plasma = FourSurf.read_focus_input(plasma_file) # I don't like this, write your own function
    
    def plasma_plot(self, zeta=0,npoints=360):
        
        rb,zb = self.plasma.rz(np.linspace(0, 2*np.pi, npoints), zeta * np.ones(npoints))
        plt.plot(rb,zb,'tab:gray',ls='--',lw=.7)
        #plt.plot(rb*100,zb*100,'tab:gray',ls='--',lw=.7)
    
