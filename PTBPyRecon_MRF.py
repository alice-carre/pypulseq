import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
from pathlib import Path

import sys
sys.path.append('C:/Users/carre01/Desktop/Python_pgm/PtbPyRecon')
import PTBRecon
import PTBRecon.Subpackages.parametric as par
import TestScripts.helper as helper

main_dir = Path('Z:/_allgemein/projects/pulseq/measurements/2022-01-26_phantom/')
meas_dir = 'meas_MID135_20220126_radial_MRF_256px_fov256_8mm_golden_angle_ohne_RFphase_FID86248'

meas_path = main_dir / meas_dir

dat_list = list(meas_path.glob('*{t}'.format(t='.dat')))
dat_file = dat_list[0]
dat_path = dat_file.parent

seq_list = list(meas_path.glob('*{t}'.format(t='.seq')))
seq_file = seq_list[0]
seq_path = dat_path

TE=4e-3
TR=7e-3
rf_phi_max =70
number_of_values = 1500
flip_angle_all = np.ones((number_of_values,))
TR_all = np.ones((number_of_values,)) * TR
TE_all = np.ones((number_of_values,)) * TE
index=0
with open(r"PTB_QUIERO_MRF.dat") as datFile:
    for data in datFile:
        flip_angle_all[index]=float(data.split()[1])*rf_phi_max
        index += 1

flip_angle_all[flip_angle_all == 0] = 0.1

def run(praw):
    ''' MRF image based dictionary matching using three different approaches to calculate the dictionary '''

    # Read in raw data
    # Create MRScan object
    mr = PTBRecon.MRScan(praw)
    mr.ReadHdr(mr)
    mr.ReadData(mr)
    ## NEW BEGIN
    mr.Pars.Recon.PulseqPathname = str(seq_path)
    mr.Pars.Recon.PulseqFilename = str(seq_file)
    mr.ReadPulseq(mr)

    #mr.NoiseDecorr(mr)

    ## NEW END
    # Sort data
    mr.SortData(mr)
    mr.Pars.Flags.IsDataSorted = 1

    mr.Pars.Recon.CoilCompNum = 4
    mr.CoilComp(mr)

    ## NEW BEGIN
    mr.CalcTraj(mr)
    mr.Pars.Recon.KDcfType = 'zwart'
    mr.CalcDcf(mr)

    mr.Pars.Recon.CsmMode = 'walsh'  # 'inati'
    mr.CalcCsm(mr)

    # Split into 1000 dynamics each with 10 lines
    mr.Pars.Recon.SplitDynSlWnd = 0
    mr.Pars.Recon.SplitDynNum = 1500
    mr.SplitDyn(mr)

    mr.Pars.Recon.KDcf = np.swapaxes(mr.Pars.Recon.KDcf, 1, 6)

    # mr.Pars.Recon.KDcfType = 'voronoi'
    # mr.CalcDcf(mr)
    # if mr.Pars.Recon.PulseqHead.get('Sampling_scheme') == 'radial' or 'spiral':
    #    mr.Pars.Recon.KDcfType = 'voronoi' #zwart, voronoi,

    # Reconstruct image data
    mr.Pars.Recon.ReconType = 'fft'
    mr.ReconData(mr)

    # Combine coil information
    mr.Pars.Recon.CoilCombineType = 'sw'
    mr.CombineCoils(mr)


    imrf = np.squeeze(mr.Data.I)

    # 2D -> 3D
    imrf = imrf[:, :, np.newaxis, :]

    # Calculate mask
    mean_im = mr.norm_im(np.abs(np.average(imrf, 3)))
    mask = np.zeros(mean_im.shape)
    mask[mean_im > 0.15] = 1

    # Load parameter

    phi = np.zeros(flip_angle_all.shape)

    # Calculate dictionary
    t1 = np.linspace(900, 2500, 60)
    t2 = np.linspace(100, 300, 60)

    mrf_im = []
    for meth in ['epg']:

        flag_dict_calculated = False
        if meth == 'bloch':
            try:
                dict_sig, dict_theta = par.mrf.calc_dict(t1, t2, flip_angle_all, TE_all, TR_all, phi, method='bloch', nf=101)
                flag_dict_calculated = True
            except:
                print('Bloch error: ', sys.exc_info())

        elif meth == 'epg':
            try:
                dict_sig, dict_theta = par.mrf.calc_dict(t1, t2, flip_angle_all, TE_all, TR_all, phi, method='epg')
                flag_dict_calculated = True

            except:
                print('EPG error: ', sys.exc_info())

        elif meth == 'bloch_slice':
            try:
                dict_sig, dict_theta = par.mrf.calc_dict(t1, t2, flip_angle_all, TE_all, TR_all, phi, method='bloch_slice', tbw=8, rf_duration=1.92, nf=101)

                flag_dict_calculated = True

            except:
                print('BlochSlice error: ', sys.exc_info())



        if flag_dict_calculated:
            # Normalise dictionary
            dict_norm, dict_norm_val = par.mrf.calc_norm_dict(dict_sig)

            # Carry out matching
            im_match, tmp = par.mrf.match_dict_3d(dict_norm, dict_theta, imrf, mask)

            # Select centre of image
            idim = im_match.shape
            im_match = im_match[idim[0] // 4:3 * idim[0] // 4, idim[1] // 4:3 * idim[1] // 4, 0, :]

            m0 = np.abs(im_match[:, :, 0] + 1j * im_match[:, :, 3])
            mrf_im.append(np.concatenate(helper.im2rgb([m0, im_match[:, :, 1], im_match[:, :, 2]], 'magma', [np.max(m0)*0.8, 2000, 300]), axis=0))

    return (mrf_im)

praw = os.path.join(dat_path, dat_file)
run(praw)
