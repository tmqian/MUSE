from FICUS import MagnetReader as mr
from FICUS import AnalyticForce as af

import numpy as np
import matplotlib.pyplot as plt
import sys

'''
This program reads the output of sample_fields.py
and evaluates forces and torques from the magnetic field

Exports a Mx30 .csv file
   head = 'Xn [m] (N face center), Yn [m], Zn [m], \
           Xs [m] (S face center), Ys [m], Zs [m], \
           Xc [m] (COM coordinate), Yc [m], Zc [m], \
           Bcx [T] (avg com B), Bcy [T], Bcz [T], \
           Fcx [N] (COM force), Fcy [N], Fcz [N], \
           Tx [N m] (Torque), Ty [N m], Tz [N m] \
           fnx [N] (torque force couple N), fny [N], fnz [N], \
           fsx [N] (torque force couple S), fsy [N], fsz [N], \
           Fnx [N] (Sum forces N), Fny [N], Fnz [N], \
           Fsx [N] (Sum forces S), Fsy [N], Fsz [N], \
           \n'

Updated 31 May 2021
'''

path = './'
f_mag   = 'block_zot80_3sq_1.csv'

try:
    f_field = sys.argv[1]
except:
    print('usage: python analysis.py f_field.csv')
    f_field = 'B-field-n2-fix2.csv'

# load field data
with open(path+f_field) as f:
    datain = f.readlines()
    
data = np.array([ line.strip().split(',') for line in datain[1:]] , float)
N_samples = len(data)
print('read data shape:', data.shape)

rx,ry,rz,bx,by,bz = data.T
bmag = np.sqrt(bx*bx + by*by + bz*bz)
Bvec = np.array([bx,by,bz]).T

# load magnet file
mag = mr.Magnet_3D(path+f_mag)

X,Y,Z = mag.com.T
s = mag.sgn

M = mag.M[0]
L = np.mean(mag.L)

Q = M*L*L

print('M', M)
print('L', L)
print('Q', Q)


# Compute forces and torques
N_magnets = mag.N_magnets
N_charges = int( N_samples / N_magnets / 2 )
print('N samples:', N_samples)
print('N magnets:', N_magnets)
print('N charges:', N_charges) # per face


B_arr = np.transpose( np.reshape(Bvec,(N_magnets,2*N_charges,3)), axes=(1,0,2) )
#B_arr = np.reshape(Bvec,(2*N_charges,N_magnets,3))
Bp = np.mean(B_arr[:N_charges],axis=0)  # positive face samples
Bm = np.mean(B_arr[N_charges:],axis=0)  # negative face samples
Bc = (Bp + Bm)/2

Fp =  Q * Bp
Fm = -Q * Bm
Fc = Fp + Fm


rvec = np.array([rx,ry,rz]).T

r_arr = np.transpose( np.reshape(rvec,(N_magnets,2*N_charges,3)), axes=(1,0,2) )
#r_arr = np.reshape(rvec,(N_charges*2, N_magnets,3))
rn = np.mean(r_arr[:N_charges],axis=0)
rs = np.mean(r_arr[N_charges:],axis=0)
rcom = (rn + rs)/2 # average

rp = rn - rcom
rm = rs - rcom

tau = np.cross(rp,Fp) + np.cross(rm, Fm)

# export forces
Fn = np.cross(tau,rp) / (np.linalg.norm(rp,axis=1)**2)[:,np.newaxis] / 2
Fs = np.cross(tau,rm) / (np.linalg.norm(rm,axis=1)**2)[:,np.newaxis] / 2

Fpn = Fc/2 + Fn
Fps = Fc/2 + Fs

print('peak field:', np.max(bmag))
print('peak face force:', np.max( np.linalg.norm(Fpn,axis=1) ))
print('peak torque:', np.max( np.linalg.norm(tau,axis=1) ))

# make plot 
plt.figure(figsize=(9,7))

plt.subplot(3,3,1)
plt.hist(np.linalg.norm(Bvec,axis=1),100, color = 'C2')
plt.title('B element [T]')

plt.subplot(3,3,2)
plt.hist(np.linalg.norm(Bp,axis=1),100, color = 'C2')
plt.title('B face [T]')

plt.subplot(3,3,3)
plt.hist(np.linalg.norm(Bc,axis=1),100, color = 'C2')
plt.title('B center [T]')

plt.subplot(3,3,4)
plt.hist(np.linalg.norm(tau,axis=1),100, color = 'C4')
plt.title('Torque [N m]')

plt.subplot(3,3,5)
plt.hist(np.linalg.norm(Fp,axis=1),100, color = 'C1')
plt.title('F face [N]')

plt.subplot(3,3,6)
plt.hist(np.linalg.norm(Fc,axis=1),100, color = 'C1')
plt.title('F center [N]')

plt.subplot(3,3,7)
# removed /2
plt.hist(np.linalg.norm(Fn,axis=1), 100, color = 'C0')
plt.title('Torque Force (+)')

plt.subplot(3,3,8)
plt.hist(np.linalg.norm(Fc/2,axis=1),100, color = 'C0')
plt.title('COM Force (+)')

plt.subplot(3,3,9)
plt.hist(np.linalg.norm(Fpn,axis=1),100, color = 'C0')
plt.title('Total Pole Force (+)')

plt.suptitle(f_field)

f_plot = f_field[:-4] + '.png'
print('saving:', f_plot)
plt.savefig(f_plot)
plt.tight_layout()
plt.draw()
plt.show()


def export_1():
    # prepare data for export
    xn,yn,zn = rn.T
    xs,ys,zs = rs.T
    xc,yc,zc = rcom.T

    fnx,fny,fnz = Fpn.T
    fsx,fsy,fsz = Fps.T
    tx, ty, tz  = tau.T
    
    data = np.array([xn,yn,zn,fnx,fny,fnz,
                     xs,ys,zs,fsx,fsy,fsz,
                     xc,yc,zc,tx,ty,tz] ).T
    
    # write
    f_write = 'ForceTorque-v6-n%i.csv' % (N_charges*2)
    with open(f_write,'w') as f:
        head = 'Xn [m], Yn [m], Zn [m], Fnx [N], Fny [N], Fnz [N], \
        Xs [m], Ys [m], Zs [m], Fsx [N], Fsy [N], Fsz [N], \
        Xc [m], Yc [m], Zc [m], Tx [N m], Ty [N m], Tz [N m] \n'
        f.write(head)
        for line in data:
            #x,y,z,fx,fy,fz = line
            out = '{:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}\n'.format(*line)
            f.write(out)
    
    print('wrote to file:', f_write)


def export_2():
    # prepare data for export
    xn,yn,zn = rn.T
    xs,ys,zs = rs.T
    xc,yc,zc = rcom.T
   
    fcx,fcy,fcz = Fc.T / 2
    bcx,bcy,bcz = Bc.T

    # torque forces
    tx, ty, tz  = tau.T
    fnx,fny,fnz = Fn.T
    fsx,fsy,fsz = Fs.T
    
    # sum of torques and COM forces
    Fnx,Fny,Fnz = Fpn.T
    Fsx,Fsy,Fsz = Fps.T
    
    data = np.array([xn,yn,zn,
                     xs,ys,zs,
                     xc,yc,zc,
                     bcx,bcy,bcz,
                     fcx,fcy,fcz,
                     tx,ty,tz,
                     fnx,fny,fnz,
                     fsx,fsy,fsz,
                     Fnx,Fny,Fnz,
                     Fsx,Fsy,Fsz
                     ] ).T
    
    # write
    f_write = 'Field-Force-Torque-v6-n%i.csv' % (N_charges*2)
    with open(f_write,'w') as f:
        head = 'Xn [m] (N face center), Yn [m], Zn [m], \
                Xs [m] (S face center), Ys [m], Zs [m], \
                Xc [m] (COM coordinate), Yc [m], Zc [m], \
                Bcx [T] (avg com B), Bcy [T], Bcz [T], \
                Fcx [N] (COM force), Fcy [N], Fcz [N], \
                Tx [N m] (Torque), Ty [N m], Tz [N m] \
                fnx [N] (torque force couple N), fny [N], fnz [N], \
                fsx [N] (torque force couple S), fsy [N], fsz [N], \
                Fnx [N] (Sum forces N), Fny [N], Fnz [N], \
                Fsx [N] (Sum forces S), Fsy [N], Fsz [N], \
                \n'
        f.write(head)
        for line in data:
            out = '{:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}, {:.6e}\n'.format(*line)
            f.write(out)
    
    print('wrote to file:', f_write)


export_2()
