import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from pypulseq import Sequence
import sys
sys.path.append('C:/Users/carre01/Desktop/Python_pgm/PtbPyRecon')
import PTBRecon
from PTBRecon import MRScan
import PTBRecon.Functions.SplitDyn as SD
from pypulseq.write_seq_definitions import read_definitions

main_dir = Path('Z:/_allgemein/projects/pulseq/measurements/2022-01-26_phantom/')
meas_dir = 'meas_MID132_20220126_radial_MRF_replace_zeros_FID86245'

meas_path = main_dir / meas_dir

dat_list = list(meas_path.glob('*{t}'.format(t='.dat')))
dat_file = dat_list[0]
dat_path = dat_file.parent

seq_list = list(meas_path.glob('*{t}'.format(t='.seq')))
seq_file = seq_list[0]
seq_path = dat_path

# Create MRScan object
mr = MRScan(os.path.join(dat_path, dat_file))

# Read header from dat file
mr.ReadHdr(mr)

# Read data from dat file
mr.ReadData(mr)

## NEW BEGIN
mr.Pars.Recon.PulseqPathname = str(seq_path)
mr.Pars.Recon.PulseqFilename = str(seq_file)
mr.ReadPulseq(mr)

## NEW END
# Sort data
mr.SortData(mr)
mr.Pars.Flags.IsDataSorted = 1

## NEW BEGIN
mr.CalcTraj(mr)
mr.CalcDcf(mr)
if mr.Pars.Recon.PulseqHead.get('Sampling_scheme') == 'radial' or 'spiral':
    mr.Pars.Recon.KDcfType = 'voronoi' #zwart, voronoi,

# Reconstruct image data
mr.Pars.Recon.ReconType = 'fft'
mr.ReconData(mr)


# Combine coil information
mr.CombineCoils(mr)

# Squeeze data and create image stack
img_stack = np.abs(np.squeeze(mr.Data.I))
img_stack = np.fliplr(np.rot90(img_stack, -1, axes=(0, 1)))

print(f'img_stack shape = {img_stack.shape}')
if len(img_stack.shape) > 2:
    fig, axs = plt.subplots(3, 6)
    for ii, ax in enumerate(axs.flatten()):
        if ii == 0:
            h = ax.imshow(img_stack[:, :, ii])
            clim = h.get_clim()
        else:
            h = ax.imshow(img_stack[:, :, ii])
            h.set_clim(clim)
else:
    fig, ax = plt.subplots()
    ax.imshow(img_stack[:, :])
plt.show()

# save as nifti
nifti_img = nib.Nifti1Image(np.squeeze(np.abs(img_stack)), np.eye(4))
nib.save(nifti_img, os.path.join(dat_path, f"{str(dat_file).split('.dat')[0]}_img-stack"))
