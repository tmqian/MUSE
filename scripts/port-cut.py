# Updated 08 January 2021

import MagnetReader as mr

import numpy as np
import sys

# usage: python port-cut.py fname.focus

### USER INPUT ###
port_diameter = 2*2.54                      # 2 inches
ports_h = [np.pi/8, 3*np.pi/8]              # toroidal angle of horizontal ports
ports_v = []                                # cm, radial displacement (from major radius) for vertical ports


f = sys.argv[1]
fd = mr.ReadFAMUS(f)

def check_horizontal(uv,u0,D,R=30.48,a=9.5):
    '''
        Given port diameter D, toroidal angle u0, 
        and toroidal radii (R,a),
        
        computes whether point (u,v) interesects port.
        Outputs boolean True if there is intersection.
    '''
    
    u,v = uv
    
    r = D/2
    A = np.arcsin( r/(R+a) )  # toroidal
    B = np.arcsin(r/a)        # poloidal
    
    p = abs(u - u0)
    
    isInside = False
    if ( p > A) :
        return isInside # False
    
    below = B*np.sqrt(1 - (p/A)**2)
    above = np.pi*2 - B*np.sqrt(1 - (p/A)**2)
    
    if (v < below) or (v > above):
        isInside = True

    return isInside
    
    
def check_vertical(uv,u0,D=2.54*2,a1=0,R=30.48,a=9.5):
    '''
        Given port diameter D, toroidal angle u0, lateral displacement a1,
        and toroidal radii (R,a),
        
        computes whether point (u,v) interesects port.
        Outputs boolean True if there is no intersection.
    '''
    
    u,v = uv
    
    # check u
    r = D/2
    A = np.arcsin( r/(R+a1) )
    p = abs(u - u0)
    
    isInside = False
    if ( p > A) :
        return isInside
    
    # check v
    t1 = np.arccos( (a1-r)/a )
    t2 = np.arccos( (a1+r)/a )
    B = (t2-t1)/2   
    v0 = (t1+t2)/2
    
    below = v0 + B*np.sqrt(1 - (p/A)**2)
    above = v0 - B*np.sqrt(1 - (p/A)**2)
    
    if ( (v < above) and (v > below) ):
        isInside = True
    
    return isInside
   
    
    
def check_ports(p):
    
    # horizonal
    for u in ports_h:
        inside =  check_horizontal(p,u,D)
        if (inside):
            return True
        
    # vertical
    for u in ports_v:
        inside =  check_vertical(p,0,D=D,a1=u)
        if (inside):
            break       
        return inside



### main
new_Ic = []
count = 0

D = port_diameter
uv = np.transpose(fd.to_uv())

for p in uv:
    inside = check_ports(p)
    if (inside):
        count += 1
        new_Ic.append(0)
    else:
        new_Ic.append(1)
new_Ic = np.array(new_Ic)
           
print('Identified %d magnet port interesections' % count) 
mag = np.sum( abs(fd.pho) * (1 - new_Ic) )
fd.pho = fd.pho * new_Ic
fd.Ic  = fd.Ic  * new_Ic # mask previous corrections
print('  removed %.1f magnets' % mag)

fd.update_data()
fout = 'port_'+f
fd.writefile(fout)
print('  wrote to file %s' % fout)
