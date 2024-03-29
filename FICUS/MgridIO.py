# this program converts binary mgrid to a netCDF mgri

import numpy as np
import netCDF4 as nc
import sys

### File I/O ###


class Read_MGRID():

    
    def __init__(self,fname='temp',binary=False,
                    nr=51,nz=51,nphi=24,nfp=2,
                    rmin=0.20,rmax=0.40,zmin=-0.10,zmax=0.10,
                    nextcur=0):

        self.nr = nr
        self.nz = nz
        self.nphi = nphi
        self.nfp = nfp

        self.rmin = rmin
        self.rmax = rmax
        self.zmin = zmin
        self.zmax = zmax

        self.n_ext_cur  = 0
        self.cur_labels = []

        self.br_arr = [] 
        self.bz_arr = [] 
        self.bp_arr = [] 

        if (binary):
            self.read_binary(fname,_debug=False)

        print('Initialized mgrid file: (nr,nphi,nz,nfp) = ({}, {}, {}, {})'.format(nr,nphi,nz,nfp))

    # takes 3D vector field (N,3) as well as coil group label (up to 30 char)
    #    input is x,y,z
    def add_field(self,B,name='default'):

        
        # structure Bfield data into arrays (phi,z,r) arrays
        bx,by,bz = B.T
        shape = (self.nr, self.nz, self.nphi) 
        bx_arr = np.reshape(bx  , shape).T
        by_arr = np.reshape(by  , shape).T
        bz_arr = np.reshape(bz  , shape).T

        # pass from cartesian to cylindrical coordinates
        phi = self.export_phi()
        #cos = np.cos(phi)[np.newaxis,np.newaxis,:]
        #sin = np.sin(phi)[np.newaxis,np.newaxis,:]
        cos = np.cos(phi)[:,np.newaxis,np.newaxis]
        sin = np.sin(phi)[:,np.newaxis,np.newaxis]
        br_arr =  cos*bx_arr + sin*by_arr
        bp_arr = -sin*bx_arr + cos*by_arr

        self.br_arr.append( br_arr )
        self.bz_arr.append( bz_arr )
        self.bp_arr.append( bp_arr )

        # add coil label
        if (name == 'default'):
            label = pad_string('magnet_%i' % self.n_ext_cur)
        else:
            label = pad_string(name)
        self.cur_labels.append(label)
        self.n_ext_cur = self.n_ext_cur + 1

    def add_field_cylindrical(self,br,bp,bz,name='default'):
        
        # structure Bfield data into arrays (phi,z,r) arrays

        self.br_arr.append( br )
        self.bz_arr.append( bz )
        self.bp_arr.append( bp )

        # add coil label
        if (name == 'default'):
            label = pad_string('magnet_%i' % self.n_ext_cur)
        else:
            label = pad_string(name)
        self.cur_labels.append(label)
        self.n_ext_cur = self.n_ext_cur + 1


    def read_netCDF(self,fname):
        
        print(' reading:', fname)
        # overwrites existing class information

        f = nc.Dataset(fname, mode='r')
        self.nr   = int( get(f, 'ir')  )
        self.nz   = int( get(f, 'jz')  )
        self.nphi = int( get(f, 'kp')  )
        self.nfp  = int( get(f, 'nfp') )
        self.n_ext_cur  = int( get(f, 'nextcur') )

        self.rmin = float( get(f, 'rmin') )
        self.rmax = float( get(f, 'rmax') )
        self.zmin = float( get(f, 'zmin') )
        self.zmax = float( get(f, 'zmax') )

        self.cur_labels = get(f, 'coil_group')


        ### read coil groups
        def unpack(binary_array):
            return "".join( np.char.decode(binary_array) ).strip()
        self.nextcur = int(f.variables['nextcur'][:])
        self.coil_names = [ unpack( self.cur_labels[j] ) for j in range(self.nextcur) ] 

        ### read raw coil current
        self.mode = f.variables['mgrid_mode'][:][0].decode()
        self.raw_coil_current = np.array(f.variables['raw_coil_cur'][:])


        print(' overwriting  mgrid coordinates: (nr,nphi,nz,nfp) = ({}, {}, {}, {})'.format(self.nr,self.nphi,self.nz,self.nfp))

    # unused
    def set_params(self,nr=51,nz=51,nphi=24,nfp=2,rmin=0.20,rmax=0.40,zmin=-0.10,zmax=0.10,nextcur=0):

        self.nr = nr
        self.nz = nz
        self.nphi = nphi
        self.nfp = nfp

        self.rmin = rmin
        self.rmax = rmax
        self.zmin = zmin
        self.zmax = zmax

        self.N_ext_cur  = 0
        self.cur_labels = []

    # retiring. Certain variable names be no longer be consistent with write()
    def read_binary(self,fname,_debug=False):
        f = open(fname,'rb')
        
        # Step 1: Read grid info (N radial points, N z points, N phi points, N field periods, N external currents)
        
        head1 = np.fromfile(f,count=4,dtype='int8') # skip header bytes, which contain array length for fortran i/o
        block1 = np.fromfile(f,count=5,dtype='int32')
        if (_debug):
            print(h1)
            #print(f.read(4)) ## print hexadecimal
            print(block1)
        nr,nz,nphi,nfp,nextcur = block1
        print('  nr, nz, nphi, nfp, n_ext_currents:', nr,nz,nphi,nfp,nextcur)
        
        # Step 2: Read grid boundary (R_min, Z_min, R_max, Z_max)
        head2 = np.fromfile(f,count=8,dtype='int8') # again skipping header bytes
        block2 = np.fromfile(f,count=4,dtype='float64')
        if (_debug):
            print(h2)
            #print(f.read(8)) ## print hexadecimal
            print(block2)
        rmin,zmin,rmax,zmax = block2 
        print('  rmin,rmax,zmin,zmax:', rmin,rmax,zmin,zmax)
        
        # Step 3: Read current groups (they come in 30 char strings)
        #.     (this could potentially be a loop)
        curlabel = []
        for j in np.arange(nextcur):
            block = np.fromfile(f,count=30,dtype='int8')
            if(_debug):
                print(block)
            msg = ''.join([chr(c) for c in block]).replace('\x00',' ').replace('\x1e',' ')
            curlabel.append(msg)
        print('  Coil Group labels: ', curlabel)
        
        # Step 4: Read B field, the encoding is Br_1, Bz_1, Bphi_1, Br_2, Bz_2 Bphi_2, ...
        npoints = nr*nz*nphi
        print('  npoints: nr * nz * nphi = ', npoints)
        head4 = np.fromfile(f,count=8,dtype='int8')
        if (_debug):
            print(head4)
            #print(f.read(8))
        
        bfield = []
        for j in np.arange(nextcur):
            head41 = np.fromfile(f,count=8,dtype='uint8')
            if (_debug):
                print(head41)
                #print(f.read(8))
            block = np.fromfile(f,count=3*npoints,dtype='float64')
            br,bz,bphi = np.reshape(block,(npoints,3)).T
            bfield.append([br,bz,bphi])
        bfield = np.array(bfield)
        
        # four extra bytes at the end, EOF?
        if (_debug):
            print(np.fromfile(f,count=4,dtype='uint8'))
        
        
        ### merge sources into single output
        #   (temporary)
        
        br,bz,bphi = np.sum(bfield, axis=0) # add the sources (input shape: M_sources, 3 components, N_points)
        
        f.close()


        ### collect outputs
        self.nr = nr
        self.nz = nz
        self.nphi = nphi
        self.nfp = nfp
        self.n_ext_cur = nextcur # rename this to N_ext_currents

        self.rmin = rmin
        self.rmax = rmax
        self.zmin = zmin
        self.zmax = zmax

        self.cur_labels = curlabel # rename this to cur_labels

        self.br = br
        self.bz = bz
        self.bphi = bphi

        # structure Bfield data from ravel into cylindrical arrays
        br_arr = np.reshape(br  , (nphi,nr,nz) )
        bz_arr = np.reshape(bz  , (nphi,nr,nz) )
        bp_arr = np.reshape(bphi, (nphi,nr,nz) )

        self.br_arr = br_arr
        self.bz_arr = bz_arr
        self.bp_arr = bp_arr



        ds = nc.Dataset(fin, 'r', format='NETCDF4')

        ds.close()

    def write(self,fout):

        ### Write
        print('Writing mgrid file')
        ds = nc.Dataset(fout, 'w', format='NETCDF4')
        
        # set dimensions
        ds.createDimension('stringsize', 30)
        ds.createDimension('dim_00001', 1)
        ds.createDimension('external_coil_groups', self.n_ext_cur)
        ds.createDimension('external_coils', self.n_ext_cur)
        ds.createDimension('rad', self.nr)
        ds.createDimension('zee', self.nz)
        ds.createDimension('phi', self.nphi)
        
        # declare variables
        var_ir = ds.createVariable('ir', 'i4')
        var_jz = ds.createVariable('jz', 'i4')
        var_kp = ds.createVariable('kp', 'i4')
        var_nfp     = ds.createVariable('nfp', 'i4')
        var_nextcur = ds.createVariable('nextcur', 'i4')
        
        var_rmin = ds.createVariable('rmin','f8')
        var_zmin = ds.createVariable('zmin','f8')
        var_rmax = ds.createVariable('rmax','f8')
        var_zmax = ds.createVariable('zmax','f8')
        
        var_coil_group = ds.createVariable('coil_group', 'c', ('external_coil_groups', 'stringsize',))
        var_mgrid_mode = ds.createVariable('mgrid_mode', 'c', ('dim_00001',))
        var_raw_coil_cur = ds.createVariable('raw_coil_cur', 'f8', ('external_coils',))
       
        
        # assign values
        var_ir[:] = self.nr
        var_jz[:] = self.nz
        var_kp[:] = self.nphi
        var_nfp[:] = self.nfp
        var_nextcur[:] = self.n_ext_cur
        
        var_rmin[:] = self.rmin
        var_zmin[:] = self.zmin
        var_rmax[:] = self.rmax
        var_zmax[:] = self.zmax
        
        var_coil_group[:] = self.cur_labels
        var_mgrid_mode[:] = 'N' # R - Raw, S - scaled, N - none (old version)
        #var_mgrid_mode[:] = 'R' # R - Raw, S - scaled, N - none (old version)
        var_raw_coil_cur[:] = np.ones(self.n_ext_cur)
        
        
        
        # go to rectangular arrays
        #cos_arr = np.cos(phi)[np.newaxis,np.newaxis,:]
        #sin_arr = np.sin(phi)[np.newaxis,np.newaxis,:]
        #
        #bx = np.ravel( br_arr*cos_arr - bphi_arr*sin_arr )
        #by = np.ravel( br_arr*sin_arr + bphi_arr*cos_arr )
        
        # transpose because binary is read (r,z,phi)
        # but netCDF is written (phi,zee,rad)

        # add fields
        for j in np.arange(self.n_ext_cur):
            
            tag = '_%.3i' % (j+1)
            var_br_001 = ds.createVariable('br'+tag, 'f8', ('phi','zee','rad') )
            var_bp_001 = ds.createVariable('bp'+tag, 'f8', ('phi','zee','rad') )
            var_bz_001 = ds.createVariable('bz'+tag, 'f8', ('phi','zee','rad') )

            var_br_001[:,:,:] = self.br_arr[j]
            var_bz_001[:,:,:] = self.bz_arr[j]
            var_bp_001[:,:,:] = self.bp_arr[j]
        
        ds.close()

        print('  Wrote to file:', fout)

    
    def init_targets(self):

        raxis = np.linspace(self.rmin,self.rmax,self.nr)
        zaxis = np.linspace(self.zmin,self.zmax,self.nz)
        
        phi   = np.linspace(0,2*np.pi/self.nfp,self.nphi)
        
        xyz = []
        for r in raxis:
            for z in zaxis:
                for p in phi:
                    x = r*np.cos(p)
                    y = r*np.sin(p)
                    xyz.append([x,y,z])
        return np.array(xyz)


    def init_target_slices(self):

        raxis = np.linspace(self.rmin,self.rmax,self.nr)
        zaxis = np.linspace(self.zmin,self.zmax,self.nz)
        
        phi   = np.linspace(0,2*np.pi/self.nfp,self.nphi)
        
        xyz = []
        for p in phi:
            s = []
            for z in zaxis:
                for r in raxis:
                    x = r*np.cos(p)
                    y = r*np.sin(p)
                    s.append([x,y,z])
            xyz.append(s)
        return np.array(xyz)


    def export_phi(self):
        phi   = np.linspace(0,2*np.pi/self.nfp,self.nphi)
        return phi

    def export_grid_spacing(self):
        return self.nr, self.nz, self.nphi


def pad_string(string):
    return '{:^30}'.format(string).replace(' ','_')


# function for reading netCDF files
def get(f,key):
    return f.variables[key][:]
