'''
    This program reads a FAMUS file, (should run shift-idx.py first)
    and a list of tower positions in (u,v) space.

    It then prints the magnet values for each tower.
    usage: python tower-print.py fname.focus list.txt
    
    Updated: 27 February 2021
'''

import MagnetReader as mr
import numpy as np
import sys

### USER INPUT ###

N_layers = 18

##################

def load_rings(self, N_poloidal=89):
    
    u,v = self.to_uv()
    v_idx = np.array(v*100,int)

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

'''
    u: i poloidal
    v: j toroidal
    w: k slice
'''
def idx_3d(u,v,w):
    return v + np.sum(rings[:u]) + N_towers*(w-1)

def idx_2d(u,v):
    return v + np.sum(rings[:u])

def show_tower(self,u,v):
    
    idx = idx_2d(u,v)
    tow = np.reshape(self.pho,(N_layers,N_towers)).T[idx]
    print('({},{}):'.format(u,v), tow)


# reads instruction set, allowing for blank lines and comments ('#')
def read_txt(fin):

    with open(fin) as f:
        datain = f.readlines()

    data = []        
    for line in datain:
        if (line.find('#') > -1): # comment
            continue
        if (not line.strip()):    # blank
            continue
        data.append( line.strip().split(',') )

    return np.array(data, int)


# load inputs
try:
    fname = sys.argv[1]
    fin = sys.argv[2]
    
    pd =  mr.ReadFAMUS(fname)
    data = read_txt(fin)

except:
    print('  usage: python tower-print.py fname.focus list.txt')
    sys.exit()

# set up coordinates
N_dipoles = len(pd.X)
N_towers  = int(N_dipoles / N_layers)
rings = load_rings(pd)

# main code
for line in data:
    u,v = line
    show_tower(pd,u,v)
