from scripts import StelloptReader as sr
import sys
import matplotlib.pyplot as plt
import numpy as np

'''
  usage: python vmec-3.py <file_tag> 

  Looking for
     wout_fname.nc

  Updated 5 April 2022
'''
f_vmec = sys.argv[1]
fname  = f_vmec[5:-3]

RADIAL = False
#RADIAL = True

print('reading vmec file: ', f_vmec)
vd = sr.readVMEC(f_vmec)

N = 200 # poloidal resolution
#s = -1  # surface index
nfp = vd.nfp
fig, axs = plt.subplots( 1,3, figsize=(12,5) )

phi = 0
sax = np.concatenate( [ [0,1], np.arange(0,vd.ns,10), [-1]])
for s in sax:

    R,Z = vd.get_surface_asym(N,phi=phi*np.pi,s=s)
    #R,Z = vd.get_surface(N,phi=phi*np.pi,s=s)
    axs[0].plot(R,Z,'C2')

axs[0].set_title(fname)

axs[0].set_aspect('equal')
axs[0].set_xlabel('R [m]')
axs[0].set_ylabel('Z [m]')
axs[0].grid()

#plt.subplot(1,3,2)

r_ax = np.linspace(0,1,vd.ns)

if RADIAL:
    rho_ax = np.sqrt(r_ax)
    rlabel = r'$\rho = \sqrt{\psi}$'
else:
    rho_ax = r_ax
    rlabel = r'$\psi$'

#axs[1].plot(vd.iota)
axs[1].plot(rho_ax, vd.iota, '.-')
axs[1].set_title('Rotational Transform')
axs[1].set_ylabel('iota',color='C0')
axs[1].set_xlabel(rlabel)
axs[1].tick_params(axis='y', labelcolor='C0')

ax2 = axs[1].twinx()
color = 'C1'
ax2.plot(rho_ax, 1/vd.iota, color+'.-')
ax2.set_ylabel('q', color=color)
ax2.tick_params(axis='y', labelcolor=color)
#axs[1,1].plot(1/ vd.iota)
#plt.plot(vd.iota)

#plt.subplot(1,3,3)
#plt.plot(vd.pressure)
axs[2].plot(rho_ax, vd.pressure/1e6,'C4.-', label='N = {:}'.format(vd.ns) )
axs[2].set_ylabel('MPa')
axs[2].set_xlabel(rlabel)
axs[2].set_title('Pressure')
axs[2].legend()
axs[2].grid()
plt.grid()

plt.tight_layout()

plt.draw()
plt.show()

