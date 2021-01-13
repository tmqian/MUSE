import StelloptReader as sr
import sys
import matplotlib.pyplot as plt

# updated 11 Jan 2021
'''
  usage: python plot_stellopt.py <file_tag>

  Looking for
     wout_fname.nc
     boozmn_fname.nc
     neo_out.fname
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
except:
    f_plasma = '../famus/estell2p.qa15.plasma'
vd.load_plasma(f_plasma)
vd.plot_vmec_3()
plt.draw()
plt.savefig('vmec_%s.png' % fname)

print('reading neo  file: ', f_neo)
plt.figure()
sr.plot_neo(f_neo,ref=True)
plt.draw()
plt.savefig('neo_%s.png' % fname)

print('reading booz file: ', f_booz)
bd = sr.readBOOZ(f_booz)
s = 46
plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
bd.plot_Booz_Contour(s_idx=s,plot_iota=True,fig=False)
plt.subplot(1,2,2)
bd.plot_B_well(s_idx=s,fig=False)
plt.draw()
plt.savefig('booz_%s.png' % fname)

plt.show()
