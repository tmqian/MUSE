#!/usr/bin/env python
# coding: utf-8

'''
    print('  usage: python3 diff-tool.py fname-1 fname-2 (optional)diff')
    last updated: 1 Jan 2021
'''
# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import sys


# In[2]:


class FAMUS_magnets():

    def __init__(self,fname):
        self.fname = fname
        self.readfile(fname)

    def readfile(self, fname):

        # open
        try:
            with open(fname) as f:
                datain = f.readlines()

        except:
            print('  error: file not found')
            sys.exit()

                # read
            
        try:
  
            data = np.array([ line.strip().split(',')[3:12] for line in datain[3:] ], float)
            info = np.array([ line.strip().split(',')[:3] for line in datain[3:] ])
            #ox,  oy,  oz,  Ic,  M_0,  pho,  Lc,  mp,  mt
            X, Y, Z, Ic, M, pho, Lc, MP, MT = np.transpose(data)
            coiltype, symmetry,  coilname = np.transpose(info)
            
            self.type = coiltype
            self.symm = symmetry
            self.name = coilname
            self.X = X
            self.Y = Y
            self.Z = Z
            self.Ic = Ic
            self.Lc = Lc
            self.M = M
            self.pho = pho
            self.MT = MT
            self.MP = MP
            
            self.data = data
            self.info = info

        except:
            print('  error: could not read .focus file')
            sys.exit()


# In[21]:

print('Welcome to FAMUS file compare')
print('  usage: python3 diff-tool.py fname-1 fname-2 (optional)diff')
#print('         the 3rd input "diff" is an optional float that sets sensitivity of comparison. Default is 1e-4.')

try:
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    f1 = FAMUS_magnets(file1)
    f2 = FAMUS_magnets(file2)
except:
    sys.exit()


# In[29]:

if (len(sys.argv) > 3):
    diff = float(sys.argv[3])
else:
    diff = 1e-4
comp = np.array(abs(f1.pho - f2.pho) > diff,int)
N_diff = np.sum(comp)
N_diff


# In[30]:

print('  reading file 1: %s' % file1)
print('  reading file 2: %s' % file2)
print('{} out of {} magnets have difference greater than {}'.format(N_diff, len(f1.pho), diff))


# In[ ]:




