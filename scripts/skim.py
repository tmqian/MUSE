import MagnetReader as mr
import numpy as np
import sys

fname = sys.argv[1]
fd = mr.ReadFAMUS(fname)

pho = fd.pho

mag = np.array( np.abs(pho) > 0, int )
print('Total dipoles:', len(pho))
print('Non-zero dipoles:', np.sum(mag))

Ndip = len(pho)

new_data = []
new_info = []

# there's probably a better way to take subset of array in python
for j in np.arange(Ndip):
    if (mag[j] == 0):
        continue
    new_data.append( fd.data[j] )
    new_info.append( fd.info[j] )

fd.data = new_data
fd.info = new_info
fd.load_data()

#print( len(fd.pho) )
fout = fname[:-6] + '_skim.focus' 
fd.writefile(fout)

