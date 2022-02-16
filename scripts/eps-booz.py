from scripts import StelloptReader as sr
import sys
import matplotlib.pyplot as plt
import numpy as np

# updated 27 August 2021
'''
  usage: python plot_stellopt.py <file_tag> <(opt)boundary.plasma>

  Looking for
     wout_fname.nc
     boozmn_fname.nc
     neo_out.fname

  Give optional 2nd file, if you want to draw a target boundary on the vmec equilibrium.
  This takes FOCUS .plasma format
'''


f_arr = sys.argv[1:]
plt.figure( figsize=(9,4) )
booz_arr = []

plt.subplot(1,2,1)
for f in f_arr:

    f_booz = 'boozmn_' + f +'.nc'

    print('reading booz file: ', f_booz)
    bd = sr.readBOOZ(f_booz)

    ax = np.linspace(0,1,len(bd.S))

    plt.plot(ax, bd.S,'.-', label=f)
    booz_arr.append( bd.S )


plt.yscale('log')
plt.ylabel(r'$| b_{mn} |_2$')
plt.legend()
plt.grid()

neo_arr = []
plt.subplot(1,2,2)
for f in f_arr:
    f_neo  = 'neo_out.'+ f
    
    print('reading neo  file: ', f_neo)
#    sr.plot_neo(f_neo,ref=False)
    nd = sr.readNEO(f_neo)
    ax = np.linspace(0,1,nd.ns)
    plt.plot(ax,nd.eps_eff, '.-', label=f)
    neo_arr.append( nd.eps_eff )

plt.yscale('log')
plt.ylabel(r'$\epsilon_{eff}^{3/2}$')
plt.legend()
plt.grid()

plt.tight_layout()

plt.figure()
j = 0
for f in f_arr:
    plt.plot(booz_arr[j], neo_arr[j], label=f)
    j += 1
plt.yscale('log')
plt.ylabel(r'$\epsilon_{eff}^{3/2}$')
plt.xscale('log')
plt.xlabel(r'$| b_{mn} |_2$')
plt.legend()
plt.grid()
plt.tight_layout()

plt.show()

import pdb
pdb.set_trace()
