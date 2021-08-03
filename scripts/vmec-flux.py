import StelloptReader as sr
import sys
import matplotlib.pyplot as plt
import numpy as np

# updated 3 August 2021
'''
  usage: python scripts/vmec-flux.py fname

  Looking for
     wout_fname.nc

'''
fname = sys.argv[1]

f_vmec = 'wout_'   + fname +'.nc'
f_booz = 'boozmn_' + fname +'.nc'
f_neo  = 'neo_out.'+ fname

print('reading vmec file: ', f_vmec)
vd = sr.readVMEC(f_vmec)

vd.plot_vmec_3( phi=[0,np.pi/4, np.pi/2] )  # default setting for NFP=2
plt.draw()
#plt.savefig('vmec_%s.png' % fname)

phipf = sr.get(vd.f,'phipf')
phi   = sr.get(vd.f,'phi')
chipf = sr.get(vd.f,'chipf')
chi   = sr.get(vd.f,'chi')

plt.figure(figsize=(9,4))

s = np.arange(len(phipf))
plt.subplot(1,3,1)
plt.plot(phi,label='phi')
plt.plot(chi,label='chi')
plt.title('flux')
plt.legend()

plt.subplot(1,3,2)
plt.plot(phipf,label='phipf')
plt.plot(chipf,label='chipf')
plt.title('flux gradient')
plt.legend()

plt.subplot(1,3,3)
plt.plot(vd.iota,label='iotaf')
plt.title('Rotation Transform')
plt.legend()

name = vd.fname
plt.suptitle(name)
plt.tight_layout()

plt.show()


