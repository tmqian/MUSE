# this program converts binary mgrid to a netCDF mgrid

import numpy as np
import netCDF4 as nc
import sys

### File I/O ###
try:
    fin = sys.argv[1]
except:
    print('usage: python mgrid-convert.py mgrid.fname')
    sys.quit()

fout = 'new-mgrid.nc'
fout = 'mgrid.FICUS' + fin[11:] + '.nc'


class Read_MGRID():

    
    def __init__(self,fname):
        self.read_binary(fname,_debug=False)


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
        self.nextcur = nextcur # rename this to N_ext_currents

        self.rmin = rmin
        self.rmax = rmax
        self.zmin = zmin
        self.zmax = zmax

        self.curlabel = curlabel # rename this to cur_labels

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


    def write(self,fout):

        ### Write
        ds = nc.Dataset(fout, 'w', format='NETCDF4')
        
        # set dimensions
        ds.createDimension('stringsize', 30)
        ds.createDimension('dim_00001', 1)
        ds.createDimension('external_coil_groups', self.nextcur)
        ds.createDimension('external_coils', self.nextcur)
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
        
        var_br_001 = ds.createVariable('br_001', 'f8', ('phi','zee','rad') )
        var_bp_001 = ds.createVariable('bp_001', 'f8', ('phi','zee','rad') )
        var_bz_001 = ds.createVariable('bz_001', 'f8', ('phi','zee','rad') )
        
        # assign values
        var_ir[:] = self.nr
        var_jz[:] = self.nz
        var_kp[:] = self.nphi
        var_nfp[:] = self.nfp
        var_nextcur[:] = self.nextcur
        
        var_rmin[:] = self.rmin
        var_zmin[:] = self.zmin
        var_rmax[:] = self.rmax
        var_zmax[:] = self.zmax
        
        var_coil_group[:] = self.curlabel
        var_mgrid_mode[:] = 'N' # R - Raw, S - scaled, N - none (old version)
        var_raw_coil_cur[:] = (1) # hard-coded, this information was not included in the binary file
        
        
        
        # go to rectangular arrays
        #cos_arr = np.cos(phi)[np.newaxis,np.newaxis,:]
        #sin_arr = np.sin(phi)[np.newaxis,np.newaxis,:]
        #
        #bx = np.ravel( br_arr*cos_arr - bphi_arr*sin_arr )
        #by = np.ravel( br_arr*sin_arr + bphi_arr*cos_arr )
        
        # transpose because binary is read (r,z,phi)
        # but netCDF is written (phi,zee,rad)
        var_br_001[:,:,:] = self.br_arr
        var_bz_001[:,:,:] = self.bz_arr
        var_bp_001[:,:,:] = self.bp_arr
        #var_br_001[:,:,:] = np.transpose(br_arr,axes=(2,1,0))
        #var_bz_001[:,:,:] = np.transpose(bz_arr,axes=(2,1,0))
        #var_bp_001[:,:,:] = np.transpose(bp_arr,axes=(2,1,0))
        
        ds.close()

        print('  Wrote to file:', fout)


mgrid = Read_MGRID(fin)
mgrid.write(fout)
