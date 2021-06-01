'''
    This program reads a FAMUS file, (should run shift-idx.py first)
    and plots a map of dipoles in (u,v) space

    usage: python mag-map.py fname.focus <idx>

    The optional index argument toggles the following output
      idx: 1-18 plots the +/- dipoles in each layer
      idx:  0 integrates over all layers, labeling each tower with positive, negative, or mixed
      idx: -1 integrates over all layers, labeling each tower highest slice's layer index
    If no <idx> is given, option -1 is selected by default.

    User Switch:
    _save: (bool) set TRUE to save .png
    _pipe: set to 'X' or 'Y' to show only x-pipe or y-pipe instead of the full half period ('none' shows both)
    _legend: set to 'L' or 'R' for positioning legend on left or right side

    Updated: 7 March 2021
'''

import MagnetReader as mr
import numpy as np
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.cm as cm

### USER INPUT ###

_save = True
_pipe = None
_legend = 'L'

###################

def slice_map(self,single_layer=-1,N_layers=18,new_fig=True, shift=False):
    
    # Plots magnet distribution, layer by layer
    u,v,projec = self.to_towers(N_layers=N_layers)
    if (shift):
        v = (v-np.pi)%(2*np.pi) - np.pi
    
    pho = self.pho
    N_towers = int(len(pho)/ N_layers)
    print(len(pho), N_towers)
    m = np.reshape(self.pho, (N_layers,N_towers))

    if (new_fig):
        plt.figure(figsize=(12,8))


    levels = np.linspace(-.9,.9,10)

    norm = mc.BoundaryNorm(levels, 256)

    if (single_layer < 0):
        for s in np.arange(N_layers):
            mag = np.argwhere( np.abs(m[s]) > 0)
            plt.plot(u[mag],v[mag],'.',label='layer %i' % (s+1) )
        if (_legend == 'L'):
            plt.legend(frameon=True,fontsize=8,loc=2)
        if (_legend == 'R'):
            plt.legend(frameon=True,fontsize=8,loc=1)

    # new option
    elif (single_layer == 0):
        above = np.sum( np.array(m>0,int), axis=0)
        below = np.sum( np.array(m<0,int), axis=0)
        both  = above*below

        idx1 = np.argwhere(above)
        idx2 = np.argwhere(below)
        idx3 = np.argwhere(both)
        plt.plot(u[idx1],v[idx1],'C3.',label='above')
        plt.plot(u[idx2],v[idx2],'C0.',label='below')
        plt.plot(u[idx3],v[idx3],'C4.',label='both')

        if (_legend == 'L'):
            plt.legend(frameon=True,fontsize=8,loc=2)
        if (_legend == 'R'):
            plt.legend(frameon=True,fontsize=8,loc=1)
    else:
        s = single_layer - 1
        above = np.argwhere(m[s] > 0)
        below = np.argwhere(m[s] < 0)
        plt.plot(u[above],v[above],'C3.')
        plt.plot(u[below],v[below],'C0.')
        plt.title('Layer %i' % (s+1) )
    plt.xlabel('toroidal half period')
    plt.ylabel('poloidal angle')

    plt.axhline(np.pi/2,ls='--',color='C1',lw=0.7)
    if (shift):
        plt.axhline(-np.pi/2,ls='--',color='C1',lw=0.7)
    else:
        plt.axhline(3*np.pi/2,ls='--',color='C1',lw=0.7)
    plt.axvline(np.pi/4,ls='--',color='C1',lw=0.7)

    
def idx_map(self,N_layers=18, new_fig=False, shift=False):
    # Plots magnet distribution, layer by layer
    u,v,projec = self.to_towers(N_layers=N_layers)
    if (shift):
        v = (v-np.pi)%(2*np.pi) - np.pi
    
    dots = load_dots(self)
    ux,vx = dots.T
    
    if (new_fig):
        plt.figure()
    us = plt.tricontour(u,v, ux,np.max(ux), colors='grey', linewidths=0.5)
    plt.clabel(us, fontsize=8, fmt='%i')
    us_heavy = plt.tricontour(u,v, ux, int(np.max(ux)/5) , colors='grey', linewidths=1.2)
    plt.clabel(us_heavy, fontsize=10, fmt='%i')
    
    vs = plt.tricontour(u,v, vx, np.max(vx), colors='y', linewidths=0.5)
    plt.clabel(vs, fontsize=8, fmt='%i')
    vs_heavy = plt.tricontour(u,v, vx, int(np.max(vx)/5) , colors='y', linewidths=1.2)
    plt.clabel(vs_heavy, fontsize=10, fmt='%i')
    
    plt.tight_layout()

def load_rings(self, N_poloidal=89):
    
    u,v = self.to_uv()
    #v = (v-np.pi)%(2*np.pi) # shift

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


def load_dots(self):
    
    rings = load_rings(self)

    dots = []
    y = 0
    for k in rings:
        for x in np.arange(k):
            dots.append((x,y))
        y += 1
        
    return np.array(dots)



fname = sys.argv[1]
pd =  mr.ReadFAMUS(fname)

try:
    s = int(sys.argv[2])
    slice_map(pd,s,shift=True)
    print('read user input: layer', s)
except:
    s = 0
    slice_map(pd,shift=True)
idx_map(pd,new_fig=False,shift=True)

fout = 'map-{}.png'.format(s, fname[:-6])
if (_pipe == 'X'):
    plt.xlim(0, np.pi/4)
    fout = 'map-x-{}.png'.format(s, fname[:-6])
if (_pipe == 'Y'):
    plt.xlim(np.pi/4, np.pi/2)
    fout = 'map-y-{}.png'.format(s, fname[:-6])

plt.title(pd.fname)
plt.tight_layout()

if (_save):
    print('  saving file:', fout)
    plt.savefig(fout)


plt.draw()
plt.show()

