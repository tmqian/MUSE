#!/usr/bin/env python
# coding: utf-8

'''
Reads (multiple) famus logs and plots convergence for BNORM, PMVOL, and DPBIN
usage: python read-famus-log.py <list-of-logs>

Last Updated: 12 Jan 2021
'''


import numpy as np
import matplotlib.pyplot as plt
import sys


def read_data(fname):
    
    # open file
    with open(fname) as f:
        datain = f.readlines()
    # locate data
    j = 0
    for line in datain:
        if (line.find('Quasi-Newton method') > 0):
            start = j+2

        if (line.find('output') > -1):
            stop = j

        if (line.find('EXITING') > 0):
            stop = j
        j+=1

    nsteps = stop - start + 1
    #print(' {:<50}: {} steps'.format(fname, nsteps) )
 
    # extract data
    data1 = datain[start:stop]
    data = np.array(  [line.strip().split(';')[1:-1] for line in data1], float )

    # optional

    chi, dE, bnorm, pmsum, dpbin, pmvol = np.transpose(data)
    print(' {:<50}, {:4d}, {:e}, {:e}, {:e}'.format(fname, nsteps, bnorm[-1], dpbin[-1], pmvol[-1] ) )
    return data


def plot_data(data,fname):
    
    chi, dE, bnorm, pmsum, dpbin, pmvol = np.transpose(data)
    
    #global fname
    tag = fname.split('/')[-1]
   
    data_arr  = [bnorm, dpbin, pmvol]
    title_arr = ['bnormal', 'dpbin', 'pmvol']
    for i in range(3):
        if (i==2):
            axs[i].plot(data_arr[i],label=tag)
        else:
            axs[i].plot(data_arr[i])
        axs[i].set_title(title_arr[i])
        axs[i].set_yscale('log')

# In[6]:


print(' usage: python read-famus-log.py <list-log-files>')

print(' {:<50}, {:4}, {:8}, {:8}, {:8}'.format('fname', 'nsteps', 'bnorm', 'dpbin', 'pmvol') )
fig, axs = plt.subplots(1, 3, figsize=(12,5) )
for fname in sys.argv[1:]:
    #print(fname)
    try:
        data = read_data(fname)
        plot_data(data,fname)
    except:
        print('  issue with file: %s' % fname)

axs[2].legend(loc=2)
#plt.suptitle('Famus Log')
#plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.tight_layout()
plt.draw() # for interactions
plt.show()

