'''
    This program reads a FAMUS file, (should run shift-idx.py first)
    and plots a map of dipoles in (u,v) space

    idx: 1-18 plots the +/- dipoles in each layer
    idx: 0 (default) integrates over all layers, labeling each tower with the outer most slice

    User Switch:
    _save: (bool) set TRUE to save .png
    _pipe: set to 'X' or 'Y' to show only x-pipe or y-pipe instead of the full half period

    Updated: 27 February 2021
'''

import MagnetReader as mr
import numpy as np
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.cm as cm

### USER SWITCH ###
_save = False
_pipe = 'Y'

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
        plt.legend(frameon=True,fontsize=8,loc=2)

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
    cs = plt.tricontour(u,v, ux,np.max(ux), colors='grey', linewidths=0.7)#,cmap=cm.Greys_r)
    plt.clabel(cs, fontsize=8, fmt='%i')
    
    vs = plt.tricontour(u,v, vx,np.max(vx), colors='y', linewidths=0.5)#,cmap=cm.Greys_r)
    plt.clabel(vs, fontsize=8, fmt='%i')
    
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

if (_save):
    plt.savefig(fout)

plt.draw()
plt.show()

