# Updated 30 October 2020

import MagnetReader as mr

import numpy as np
import sys

import matplotlib.pyplot as plt

f = sys.argv[1]
fd = mr.ReadFAMUS(f)

fd.plot_symm()

fout = 'map_%s.png'%f
plt.savefig(fout)
print('Wrote magnet map to %s'%fout)

