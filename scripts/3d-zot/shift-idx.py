'''
    This program reads a FAMUS file, with p89.1 index notation
    and shifts the indices saving a new file which follows

    toroidal idx (0-33, or 0-84)
    poloidal idx (0-89)
    layers (0-17)  
'''

import MagnetReader as mr
import numpy as np
import sys

def shifted_ring_count(self, N_poloidal=89):
    
    u,v = self.to_uv()
    v = (v-np.pi)%(2*np.pi) # shift

    v_idx = np.array(v*100,int)
    N_towers = 5514

    rings = []
    j = 0
    while j < N_towers:
        
        count = 1
        while ( np.abs(v[j+1] - v[j]) < 1./N_poloidal):
            count += 1
            j += 1
            
        rings.append(count)
        j += 1
        
    return rings

def toroidal_idx(j,rings):
    Nz = rings[j]
    bump = np.sum( rings[:j] )

    temp_x = np.arange(Nz/2)
    temp_y = Nz - temp_x - 1

    ring_idx = np.concatenate((temp_x,temp_y)) + bump
    return ring_idx

def slice_idx(t_arr, N_layers=18):
    s_arr = []
    for j in np.arange(N_layers):
        bump = j*len(t_arr)
        s_arr.append(t_arr + bump)

    w_arr = np.array(np.concatenate(s_arr))
    return w_arr

def whole_idx(self, N_poloidal=89):
    
    # calculates (shifted) indices for each ring (1D)
    rings = shifted_ring_count(self, N_poloidal=N_poloidal)
    
    # creates an array for each slice (2D)
    p_arr   = np.array( (np.arange(N_poloidal) - N_poloidal/2) % N_poloidal, int )
    sub_arr = [toroidal_idx(j,rings) for j in p_arr]
    t_arr   = np.array(np.concatenate(sub_arr), int)
    
    # computes index for the whole device (3D)
    whole_arr = slice_idx(t_arr)
    return whole_arr


fname = sys.argv[1]
pd =  mr.ReadFAMUS(fname)

w_idx = whole_idx(pd)
pd.data = pd.data[w_idx]
pd.load_data()

fout = fname[:-6] + '-shift.focus'
pd.writefile(fout)
