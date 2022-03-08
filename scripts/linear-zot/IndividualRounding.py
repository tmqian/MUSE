import numpy as np
from FICUS import MagnetReader as mag
import matplotlib as mpl
from coilpy import focushdf5
from netCDF4 import Dataset
import sys 

Save = True # True or False if you want to save file at the end. Name output file at the end

fname = sys.argv[1]
#fname = 'Test1-FoldingPhoGlobal_com_zot80-m.focus'
dist = mag.ReadFAMUS(fname)

fin = sys.argv[2]
info = focushdf5.FOCUSHDF5(fin)
f = Dataset(fin, mode='r')
print('Current chi2b from inputed h5 file is:', f.variables["chi2b"][:])

# turn the 1/2 period PMs into a full torus to calculate chi2b

#dist.skim() # dont skim if you want to optimize later 
dist.halfperiod_to_fulltorus()

pho = dist.pho
ArgPartial = np.argwhere((np.abs(pho) !=1)*(np.abs(pho)!=.0001)*(pho != 0))

print('Out of', len(pho)/4, 'there are ', len(ArgPartial)/4, 'partially filled slices per 1/2 period')
print('Calculating Inductance Matrix: Takes ~5min...')

#################################### Calculate the Inductance Matrix

# magnetic moment variables: array formated as one per magnet 
pho = dist.pho
M = dist.M

# magnetic moment orientation 
MP = dist.MP
MT = dist.MT    # plot to see if this is bugged and needs to be shifted by np.pi/2b (it is in earlier grids)
                # or look at mrr = np.sqrt(mnx*mnx+mny*mny+mnz*mnz) it all should be one

mnz = np.cos(MT)
mnx = np.sin(MT) * np.cos(MP)
mny = np.sin(MT) * np.sin(MP)
mr = np.transpose([mnx,mny,mnz])

# Current Bn array of run we are looking at 

OgBn = np.ravel(info.Bn)


# Normal Vector to plasma 
nx = info.nx
ny = info.ny
nz = info.nz
xs = info.xsurf
ys = info.ysurf
zs = info.zsurf

N = np.transpose([np.ravel(nx),np.ravel(ny),np.ravel(nz)]) 

# compute distances from PM to plasma 
xm = dist.X
ym = dist.Y
zm = dist.Z
rm = np.transpose([xm,ym,zm])
rs = np.transpose( [np.ravel(xs), np.ravel(ys), np.ravel(zs)] )

dr = rm[np.newaxis,:] - rs[:,np.newaxis]

# magnitudes of distances for inductance matrix 
normdr = np.linalg.norm(dr,axis=2)

rm5 = 1/(normdr**5)
rm3 = 1/(normdr**3)

# constants 
Con = 1*10**-7  # 4pi's cancel, in units of H/m 


nax = np.newaxis

normN = np.linalg.norm(N,axis=1) # Normal unit vector 

rdotN=np.sum((dr*N[:,nax,:]),axis=2) # dot product in first term 

print('Almost done!')

gmatrix = ((rdotN*rm5*3)[:,:,nax]*dr - N[:,nax,:]*rm3[:,:,nax])*Con  # full inductance matrix 

G = np.sum(gmatrix*mr*M[:,nax],axis=2)  # inductance matrix with PM orientation and strength built in, all that is missing is pho

Bn_pm = np.sum(G*pho, axis = 1)  # final Bn matrix summed along the PM axis 
Bn_pm = Bn_pm/(normN)   # this is for show, it is just dividing by one in my experience    


#################################### Calculating chi2b as a refrence 

fin = 'tf-coils-pg19-halfp.h5'

info = focushdf5.FOCUSHDF5(fin)
f = Dataset(fin, mode='r')


Bn_coil  = np.ravel(info.Bn)
jacobian = np.ravel(info.nn)
N_theta = info.Nteta
N_zeta = info.Nzeta

Bn_tot = (Bn_pm + Bn_coil)


refBnorm = np.sum(Bn_tot**2*np.abs(jacobian)) * np.pi**2/(N_theta*N_zeta)/2

print('Refrence chi2b:',refBnorm)



##################################### Folding Pho: full torus PM array is needed to calculate accuarate chi2b, but to ensure symmetry we only want to optimize a 1/2 period array of PM magnets. Solution: fold the array into a half period so changes are symmetric 


# since we will be adding the effect of symmetric magnets in seperate quadrants, we need the sign information from pho to go into the inductance matrix 

np.seterr(invalid='ignore')

phosign = np.abs(pho)/pho
phosign[np.isnan(phosign)]=0 # turn nan (from 0/0) to zero 

Gsign = G * phosign # inductance matrix with sign information 

# next reshape

N = int(len(pho)/4)

Gnew = np.reshape(Gsign,(N_theta*N_zeta,4,N)) # keeps plasma dimesnion intact 

Pho = np.reshape(pho,(4,N))
Pho = Pho[0,:]
PhoSign = np.abs(Pho)/Pho
PhoSign[np.isnan(PhoSign)]=0
PhoSignless = PhoSign*Pho
PhoSignless # this is the pho array without sign for our new half period shape 

# sidenote you can check that the total Bn is still the same, and look at the contribution from each quadranrt
# np.sum(Gnew[:,0,:]) vs np.sum(Gnew[:,1,:]) vs np.sum(Gnew[:,2,:]) vs np.sum(Gnew[:,3,:])
# and
# np.sum(G*pho) = np.sum(Gnew*PhoSignless) ## correct!

Gnew = np.sum(Gnew,axis =1) # sum each of the 4 symetrical PM into one effect on the plasma 



################################# Main Tool 

def check_bnorm_change(index, change, refbnorm):
    
       
    Jold = Gnew[:,index]*pho[index]
    Jnew = Gnew[:,index]*change
  
    # calculate chi2b
    
    Bn_tot = (Bn_pm - Jold[:,0] + Jnew[:,0] + Bn_coil)
    
    bnorm = np.sum(Bn_tot**2*np.abs(jacobian)) * np.pi**2/(N_theta*N_zeta)/2
    
    # %change

    PerChan = (bnorm - refbnorm)/refbnorm
    
    
    
    return PerChan*100




ArgPartial = np.argwhere((np.abs(pho) !=1)*(np.abs(pho)!=.0001)*(pho != 0))

# where 

pho = PhoSignless

############################## Main Loop 

with open('IndividualCounter.txt', 'w') as f:
        f.write('Hello Welcome to Individul Rounding: \n We have {} out of {} magnets to optimize'.format(len(ArgPartial),len(pho)))


Iter = True # Turn on if you want multiple loops

TotalcounterUp = 0
TotalcounterDown = 0
TotalcounterNull = 0 
TotalIter = 1 

while Iter == True: 

    counterUp = 0
    counterDown = 0
    counterNull = 0 

    # calculate global values
    ArgPartial = np.argwhere((np.abs(pho) !=1)*(np.abs(pho)!=.0001)*(pho != 0))

    #baseline 


    for i in range(len(ArgPartial)):

        if i > 0:  # after first decision update Bn_pm with the effect of new changes 

            Jbas = Gnew[:,ArgPartial[i-1]]*phoT       # old column 
            Jopt = Gnew[:,ArgPartial[i-1]]*ChosenPho  # new chosen 

            Bn_pm = Bn_pm - Jbas[:,0] + Jopt[:,0]  # updated matrix 

        #test changing pho 

        phoT = pho[ArgPartial[i]]

        ChangeUp = check_bnorm_change(ArgPartial[i], 1 , refBnorm )

        ChangeDown = check_bnorm_change(ArgPartial[i], 0 , refBnorm )

        # sign info for labeling 


        if ChangeDown < 0 and ChangeDown < ChangeUp:

            with open('IndividualCounter.txt', 'a') as f:
                f.write('\n You should round Pm # {} from {} to 0: chi2b changed by {}%'.format(ArgPartial[i], phoT*PhoSign[ArgPartial[i]], ChangeDown))


            ChosenPho = 0
            counterDown = counterDown + 1

        elif ChangeUp < 0 and ChangeUp < ChangeDown:
                
            with open('IndividualCounter.txt', 'a') as f:    
                f.write('\n You should round Pm #{} from {} to {}: chi2b changed by {}%'.format(ArgPartial[i],phoT*PhoSign[ArgPartial[i]],PhoSign[ArgPartial[i]],ChangeUp ))
                
            ChosenPho =  1
            counterUp = counterUp + 1 

        else: 
                        
            with open('IndividualCounter.txt', 'a') as f:
                f.write('\n Leaving Pm # {} alone'.format(ArgPartial[i]))
            ChosenPho = phoT
            counterNull = counterNull + 1
   
        #update pho 
        pho[ArgPartial[i]] = ChosenPho
                
    print('Loop Number:',TotalIter)
    print(counterUp,   'PMs rounded up')
    print(counterDown, 'PMs rounded down')
    print(counterNull, 'Pms stayed the same')
   
        
    TotalcounterUp = TotalcounterUp + counterUp 
    TotalcounterDown = TotalcounterDown + counterDown
    TotalcounterNull = TotalcounterNull  
    TotalIter = TotalIter + 1 
    
    if counterUp and counterDown == 0:                         # I want this to end while loop, but it is bugged     
        print("All Done")
        print('Total iterations:', TotalIter)
        print('Total PMs rounded up:', TotalcounterUp)
        print('Total PMs rounded down:', TotalcounterDown)
        print('Total PMs unchanged:', TotalcounterNull)
        Iter = False
    if TotalIter == 30:                                        # This temperarily ends while loop
        Iter = False


####################################






## get symmetry back 

## full
x = pho*PhoSign

Fullpho = np.concatenate((x,-x,x,-x))


## new chi2b
Bn_pmT = np.sum(G*Fullpho, axis = 1)
Bn_pmT = Bn_pmT/(normN)
Bn_tot = (Bn_pmT + Bn_coil)

Bnorm = np.sum(Bn_tot**2*np.abs(jacobian)) * np.pi**2/(N_theta*N_zeta)/2
print('New chi2b:',Bnorm)

## save 

if Save == True:
    dist = mag.ReadFAMUS(fname)

    dist.pho = x # saving as a 1/2 period focus file 
    fout = 'LinearOpt_' + fname 
    dist.writefile(fout)
