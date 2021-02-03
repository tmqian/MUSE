# Updated 12 January 2021

import MagnetReader as mr

import numpy as np
import sys

# usage: python slice-layer.py fname.focus

### USER INPUT ###
set_layer  = [1,1,1,1,
              1,1,1,1,
              1,1,1,1,
              1,1,1,1,
              0,0]

### main

# open file
f = sys.argv[1]
fd = mr.ReadFAMUS(f)

# find the layers
N_mag = len( fd.pho )
N_layers = len(set_layer)
N_towers = N_mag / N_layers
print( '  {} total magnets'.format(N_mag) )
print( '  user specified {} layers'.format(N_layers) )
print( '  found {} magnets per layer'.format(N_towers) )

if (N_towers == int(N_towers)):
    N_towers = int(N_towers)
else:
    print('  check N layers')
    sys.exit()

# make mask
arr = []
for j in np.arange(N_layers):
    block = np.ones(N_towers) * set_layer[j]
    arr.append(block)

new_Ic = np.ndarray.flatten(np.array(arr))
fd.Ic  = fd.Ic  * new_Ic
fd.pho = fd.pho * new_Ic
fd.update_data()
fd.writefile('slice_'+f)

print('  applied mask: {}'.format(set_layer) )
print('  wrote to file: slice_{}'.format(f) )
