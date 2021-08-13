from scripts import MagnetReader as mr
import numpy as np
import sys

'''
    Gives a quick plot of PM array, looking down from above.
    Reads 3D magnet file.

    Updated 13 August 2021
'''

import matplotlib.pyplot as plt

f_focus = sys.argv[1]
try:
    mag = mr.Magnet_3D_gen(f_focus)
except:
    mag = mr.Magnet_3D_gen(f_focus, HLW=True)

x,y,z = mag.r0.T

plt.figure()
plt.plot(x,y,'.')
plt.axis('equal')
plt.show()
