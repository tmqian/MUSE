'''
   usage: python famus-launch.py batch.setup
   see scripts/famus-launch/ directory for sample input files

   The program reads a table and sample files 
   to setup a batch of .input and .sh

   The table lives in batch_name.setup.
   Read sample.input and b-sample.sh 

   Updated: 4 Feb 2021
'''

import sys
import numpy as np

path = './scripts/famus-launcher/'
f_sample = path+'sample-mgrid.input'
b_sample = path+'b32-sample.sh'
N_shell  = 1

### load (optional) user input
def load_option(args):
    #print('additonal options:', args)
       
    if (args[0] == '-f'):
        print(' detected user input -f')
        global f_sample
        f_sample = args[1]

    if (args[0] == '-b'):
        print(' detected user input -b')
        global b_sample
        b_sample = args[1]

    if (args[0] == '-s'):
        print(' detected user input -s')
        global N_shell
        N_shell = int(args[1]) 
       
    try:
        load_option(args[2:])
    except:
        pass

def load(fname):
    try:    
        with open(fname) as f:
            sample_input = f.readlines()
        print(' reading sample: ', fname)
    except:
        print(' sample input \'%s\' not found' % fname)
        exit()

    return sample_input    


def load_setup(fin):
    # reads a .setup file and splits lines, removing white spaces
    with open(fin) as f:
        input_data = f.readlines()

    indata = []
    for line in input_data:
        if (line.find('#') > -1):
            continue
        data   =  line.strip().split(',') 
        values = [x.strip() for x in data]
        indata.append(values)

    return np.array(indata)


def write_input(new_input, value):

    with open(new_input, 'w') as f:
        for line in sample_input:

            for j in np.arange(N_edits):
                knob = input_knobs[j]
                if (line.find( knob ) > -1):
                    line = ' {:15} =   {},\n'.format(knob, value[j])
                    #print('  %s'%line)
            f.write(line)
        print('  writing {:40}: {}'.format(new_input,value) )


def write_batch(new_batch, tag):

    #tag = '.'.join(new_input.split('.')[:-1])
    with open(new_batch, 'w') as f:
        for line in sample_batch:
           f.write(line)

        cmd = 'srun ~/CODE/FOCUS/bin/xfocus {}.input > {}.log'.format(tag,tag)
        f.write('\n{}\n'.format(cmd) )
    
    print('  writing {}'.format(new_batch) )
 
def append_batch(open_batch, tag):

    with open(open_batch, 'a') as f:

        cmd = 'srun ~/CODE/FOCUS/bin/xfocus {}.input > {}.log'.format(tag,tag)
        f.write('{}\n'.format(cmd) )
    
    print('    updating {}'.format(open_batch) )




# load options
if (len(sys.argv) > 2):
    try:
        load_option( sys.argv[2:] )
    except:
        print('usage: python famus-launch.py batch.setup -f alt-opt.input -b alt-opt.sh')
    
sample_input = load(f_sample)

#fin = sys.argv[1]
#batch_name = '.'.join(fin.split('.')[:-1])
#print(' batch name:', batch_name)
#indata = load_setup(fin)
#target_name = indata[0][0]
#input_knobs = indata[0][1:]
input_knobs = ['INPUT_COILS']

# write inputs
N_edits = len(input_knobs)
files = sys.argv[1:]

batch_name='mgrid_%i' % len(files)
print(' batch name: {:37}: {}'.format(batch_name, input_knobs) )
for fname in files:
#for line in indata[1:]:
    #tag   = line[0]
    edits = [fname]
    #edits = line[1:]

    new_input = fname[:-6] + '-m.input'
    #new_input = '{}-{}-{}.input'.format(target_name, batch_name, tag) 
    write_input(new_input, edits)


# write batch
sample_batch = load(b_sample)
f_script = 'run-{}.sh'.format(batch_name)
fsh = open(f_script,'w')

#tags = files
#tags = indata[1:,0]
#for j in np.arange( len(tags) ):
for j in np.arange( len(files) ):
    fname = files[j]
    #handle = '{}-{}-{}'.format(target_name, batch_name, tag) 
    handle = fname[:-6] + '-m'

    recycle = j % (N_shell)
    if (recycle):
        append_batch(new_batch, handle)
        continue

    #new_batch = 'b-{}-{}.sh'.format(batch_name, tag) 
    new_batch = fname[:-6] + '-m.sh'
    write_batch(new_batch, handle)
    fsh.write('sbatch %s\n' % new_batch)


print(' exporting script: {}'.format(f_script) )

