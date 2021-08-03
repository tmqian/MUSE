import StelloptReader as sr
import sys
import matplotlib.pyplot as plt
import numpy as np

# updated 5 February 2021
'''
  usage: python plot_stellopt.py <file_tag> <(opt)boundary.plasma>

  Looking for
     wout_fname.nc
     boozmn_fname.nc
     neo_out.fname

  Give optional 2nd file, if you want to draw a target boundary on the vmec equilibrium.
  This takes FOCUS .plasma format
'''
fname = sys.argv[1]

f_vmec = 'wout_'   + fname +'.nc'
f_booz = 'boozmn_' + fname +'.nc'
f_neo  = 'neo_out.'+ fname

print('reading vmec file: ', f_vmec)
vd = sr.readVMEC(f_vmec)
try:
    f_plasma = sys.argv[2]
    print('  loading plasma file: ', f_plasma)
    vd.load_plasma(f_plasma)
except:
    #f_plasma = '../famus/estell2p.qa15.plasma'
    f_plasma = 'PG2p.qa19.plasma'

vd.plot_vmec_3()
#vd.plot_vmec_3( phi=[0,np.pi/4, np.pi/2] )  # default setting for NFP=2
plt.draw()
plt.savefig('vmec_%s.png' % fname)

print('  plotting iota profile')
plt.figure()
vd.plot_iota(fig=False,ref=True)
plt.draw()
plt.savefig('iota_%s.png' % fname)

print('reading neo  file: ', f_neo)
plt.figure()
sr.plot_neo(f_neo,ref=True)
plt.draw()
plt.savefig('neo_%s.png' % fname)

print('reading booz file: ', f_booz)
bd = sr.readBOOZ(f_booz)
s = 23
plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
bd.plot_Booz_Contour(s_idx=s,plot_iota=True,fig=False)
plt.subplot(1,2,2)
bd.plot_B_well(s_idx=s,fig=False)
plt.draw()
plt.savefig('booz_%s.png' % fname)

plt.show()
