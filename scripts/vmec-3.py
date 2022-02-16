from scripts import StelloptReader as sr
import sys
import matplotlib.pyplot as plt
import numpy as np

'''
  usage: python vmec-3.py <file_tag> 

  Looking for
     wout_fname.nc

  Updated September 28, 2021
'''
fname = sys.argv[1]

f_vmec = 'wout_'   + fname +'.nc'

print('reading vmec file: ', f_vmec)
vd = sr.readVMEC(f_vmec)

N = 50
nfp = vd.nfp
s = -1
plt.figure()
for phi in [0, 1/nfp/2, 1/nfp]:
    R,Z = vd.get_surface(N,phi=phi*np.pi,s=s)
    if phi > 0:
        tag = r'$\pi$/{}'.format(int(1/phi))
    else:
        tag = 0
    plt.plot(R,Z,label=tag)

plt.title(fname)

#sr.plot_circle( R=vd.Rmajor, a=vd.Aminor )
sr.plot_circle( R=0.305, a=0.076 )
#vd.plot_vmec_3( phi=[0,np.pi/4, np.pi/2] )  # default setting for NFP=2
plt.legend()
plt.axis('square')

plt.xlabel('R [m]')
plt.ylabel('Z [m]')

plt.draw()
plt.show()

