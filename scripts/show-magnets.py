# Updated 5 January 2021

import MagnetReader as mr

import numpy as np
import sys

import matplotlib.pyplot as plt

f = sys.argv[1]
fd = mr.ReadFAMUS(f)

fd.plot_symm()

plt.title(f)
plt.draw() # for interactions
plt.show()

fout = 'map_%s.png'%f
plt.savefig(fout)
print('Wrote magnet map to %s'%fout)
