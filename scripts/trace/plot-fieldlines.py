import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from netCDF4 import Dataset
from coilpy import *
import sys

# usage: python plot-fieldlines.py file.h5 boundary.plasma
# updated: 21 Jan 2021
'''
inputs
1: .h5 file from XFIELDLINES
2: .plasma boundary from FAMUS
'''

fin = sys.argv[1]
_show = True

try:
    fboundary = sys.argv[2]
except:
    fboundary = 'PG2p.qa19.plasma'
#fboundary = 'ncsx_2p_vacuum.plasma'

def plasma_plot(zeta=0,npoints=360):
    rb,zb = plasma.rz(np.linspace(0, 2*np.pi, npoints), zeta * np.ones(npoints))
    plt.plot(rb,zb,'tab:gray',ls='--',lw=.7)
    
    
def field_plot(phi=0):
    angle = phi*phiaxis[-1]/np.pi/npoinc

    for surf in np.arange(0,nlines):
        r = r_lines[phi::npoinc,surf]
        z = z_lines[phi::npoinc,surf]

        if (surf%12 == 0):
            plt.plot(r,z,'.',color=cmap[surf], ms=1,label='{} / {}'.format(surf,nlines) )
        else:
            plt.plot(r,z,'.',color=cmap[surf], ms=1)


    plt.title(r'$\varphi$ = %.2f $\pi$' %angle)
    plt.xlim(0.22,0.4)
    plt.legend(loc=2,fontsize=8)
    plasma_plot(angle*np.pi)

    plot_circle()
    plt.axis('equal')
    plt.xlim(.22,.38)


def plot_circle(R=0.3048,a=0.0762,N=100):
    tx = np.linspace(0,np.pi*2,N)
    
    r = [R + a*np.cos(t) for t in tx]
    z = [a*np.sin(t) for t in tx]
    
    plt.plot(r,z,'k--')

def get(f,key):
    return f.variables[key][:]


# get data
f = Dataset( fin, mode='r' )
r_lines = get(f,'R_lines')
z_lines = get(f,'Z_lines')
phiaxis = get(f,'phiaxis')
npoinc = get(f,'npoinc')[0]
nlines = get(f,'nlines')[0]
plasma = FourSurf.read_focus_input(fboundary)

# make plots
temp = cm.rainbow(np.linspace(0, 1, 64))
cmap = np.concatenate((temp,temp[::-1]))

plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
field_plot(phi=0)
plt.subplot(1,3,2)
field_plot( phi=int(npoinc/4) )
plt.subplot(1,3,3)
field_plot( phi=int(npoinc/2) )

title = fin[11:-3]
plt.suptitle(title)

if (_show):
    plt.draw()
    plt.show()

# save
fout = 'trace-%s.png' % title
plt.savefig(fout)
print('wrote to file: %s' % fout)


