import math
import copy
import numpy as np
from matplotlib import pyplot as plt

import pypulseq as pp
from pypulseq.gradient_distort_GIRF import gradient_distort_GIRF
from pypulseq.gradient_distort_GIRF import plot_grad_predicted_measured


import os
import nibabel as nib
from pathlib import Path
from pypulseq import Sequence
import sys

sys.path.append('C:/Users/carre01/Desktop/PtbPyRecon')
import PTBRecon
from PTBRecon import MRScan


# load data from echo
praw = r'Z:/_allgemein/projects/Katja/forOthers/forChristophK/girf/'
# praw = ''

# load data
triangles_meas_load = -(np.load(praw + 'triangles_meas.npy'))   # load with get_meas_data(), -: match with input
triangles_meas = triangles_meas_load.swapaxes(2, 1)
input_triangle = np.load(praw + 'input_triangle.npy')
girf = np.load(praw + 'girf.npy')
freq_range_zp = np.load(praw + 'freq_range_zp.npy')

main_dir = Path('Z:/_allgemein/projects/pulseq/measurements/2021-12-13_phantom/')
meas_dir = 'meas_MID196_Vds_spiral_1_shot_oversampling_16_NX_192_FID85144'
meas_path = main_dir / meas_dir

dat_list = list(meas_path.glob('*{t}'.format(t='.dat')))
dat_file = dat_list[0]
dat_path = dat_file.parent

seq_list = list(meas_path.glob('*{t}'.format(t='.seq')))
seq_file = seq_list[0]
seq_path = dat_path


# Create MRScan object
mr = MRScan(os.path.join(dat_path, dat_file))

# Create Pulseq Sequence object
seq = Sequence()
seq.read(os.path.join(seq_path, seq_file))

for e in seq.shape_library.data:
    #the Gx and Gy gradient-shapes are e=3 and e=4
    grad_distorted, grad_distorted_zeros = gradient_distort_GIRF(grad_input=seq.shape_library.data[e][1:], grt=10, h=girf,
                                                                f_range=freq_range_zp, nb_zeros=20)
    t_in = np.arange(0, len(seq.shape_library.data[e][1:])) * 10 * 1e-3  # [ms], [dt=10us]
    plt.figure()
    plt.plot(t_in, seq.shape_library.data[e][1:])
    plt.plot(t_in, grad_distorted[:, 1])




