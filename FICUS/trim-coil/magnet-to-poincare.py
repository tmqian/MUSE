import os
import sys
import fnmatch
import time

'''
This code takes a csv trim coil magnet file as an input and outputs Poincare plots in pdf form

Updated 12 August, 2021
'''

os.system("module load mod_stellopt; module load stellopt")


#### get the files and directories
fname = sys.argv[1]
tag = fname[:-4]
vmec_directory = "vmec_inputs"
magnet-to-poincare-code_path = "/magnet-to-poincare-code"

#### put the .csv file through write-mgrid
os.system("srun -n 32 -t 5:00:00 --mem-per-cpu=8GB python ~{}/write-mgrid2.py {}".format(magnet-to-poincare-code_path, fname))


#### wait for write-mgrid to finish
while(True):
        if int(os.popen("squeue -u dseidita | less | grep -c 'python'").read()) == 0:
                break
        time.sleep(600)


#### reconfigure the netCDF file to work with this version of HDF5 - only necessary on Portal
#os.system("nccopy -k nc6 {}.nc {}_.nc".format(tag, tag))

#### iterate across all the vmec input files with the generated .nc file to get the fieldlines
for file in os.listdir(vmec_directory):
	os.system("python ~/magnet-to-poincare-code/run-fieldline2.py {}_.nc {}/{}".format(tag, vmec_directory, file))
	time.sleep(5)


#### wait for all of the slurm jobs to run before moving on
while(True):
	if int(os.popen("squeue -u dseidita | less | grep -c 'xfield'").read()) == 0:
		break
	time.sleep(10)

#### make a directory for the pdf's
os.mkdir("{}_pdfs".format(tag))


#### Create Poincare plots for all of the fieldlines and save them as pdf's
for file in os.listdir("./"):
	if fnmatch.fnmatch(file, '*.h5'):
		os.system("python ~/magnet-to-poincare-code/plot-fieldlines2.py {} ~/magnet-to-poincare-code/PG2p.qa19.plasma".format(file))





