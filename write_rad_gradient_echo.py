import math
import copy
import numpy as np
from matplotlib import pyplot as plt


import pypulseq as pp
from pypulseq.rotate import rotate
from pypulseq.rotate import scale_grad
from pypulseq.rotate import get_grad_abs_mag

seq = pp.Sequence()  # Create a new sequence object
fov = 250e-3  # Define FOV
Nx = 256  # Define resolution
alpha = 10  # flip angle
slice_thickness = 3e-3  # slice
TE = np.array([8e-3])  # TE; give a vector here to have multiple TEs (e.g. for field mapping)
TR = np.array([100e-3])  # only a single value for now
Nr = 128  # number of radial spokes
Ndummy = 20  # number of dummy scans
delta = np.pi / Nr  # angular increment

# Golden Angle Case:
#delta = (np.pi / 180) * (180 * 0.618034) # angular increment # full spokes

# more in-depth parameters
rf_spoiling_inc = 117  # RF spoiling increment
system = pp.Opts(max_grad=28, grad_unit='mT/m', max_slew=80, slew_unit='T/m/s', rf_ringdown_time=20e-6,
                 rf_dead_time=100e-6, adc_dead_time=10e-6)
rf, gz, gzr = pp.make_sinc_pulse(flip_angle=alpha * np.pi / 180, duration=4e-3, slice_thickness=slice_thickness,
                                 apodization=0.5, time_bw_product=4, system=system, return_gz=True)

# Define other gradients and ADC events
delta_k = 1 / fov
gx = pp.make_trapezoid(channel='x', flat_area=Nx * delta_k, flat_time=6.4e-3, system=system)
adc = pp.make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system)
adc.delay = adc.delay - 0.5 * adc.dwell  # compensate for the 0.5 samples shift

gx_pre = pp.make_trapezoid(channel='x', area=-gx.area / 2, duration=2e-3, system=system)
gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, duration=2e-3, system=system)

# Gradient spoiling
gx_spoil = pp.make_trapezoid(channel='x', area=0.5 * Nx * delta_k, system=system)
gz_spoil = pp.make_trapezoid(channel='z', area=4 / slice_thickness, system=system)

# Calculate timing
delay_TE = np.ceil((TE - pp.calc_duration(gx_pre) - gz.fall_time - gz.flat_time / 2 - pp.calc_duration(
    gx) / 2) / seq.grad_raster_time) * seq.grad_raster_time
delay_TR = np.ceil((TR - pp.calc_duration(gx_pre) - pp.calc_duration(gz) - pp.calc_duration(
    gx) - delay_TE) / seq.grad_raster_time) * seq.grad_raster_time

assert np.all(delay_TR >= pp.calc_duration(gx_spoil, gz_spoil))

rf_phase = 0
rf_inc = 0

for i in range(-Ndummy, Nr + 1, 1):
    for j in range(0, len(TE)):
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
        seq.add_block(rf, gz)
        phi = delta * (i - 1)

        # Rotation of gradient object(s) about the given axis and projection on cartesian axis to add it as a block to the sequence
        # Not very clean way to add events but the best way I found since add_block wouldn't take a list[SimpleNameSpace] as argument
        #The list returned by the rotate function has a variable length since sometimes gradients have to be projected on one or two axis

        rot1 = rotate('z', phi, gx_pre, gz_reph)
        if len(rot1) == 1:
            seq.add_block(rot1[0])
        elif len(rot1) == 2:
            seq.add_block(rot1[0], rot1[1])
        elif len(rot1) == 3:
            seq.add_block(rot1[0], rot1[1], rot1[2])
        elif len(rot1) == 4:
            seq.add_block(rot1[0], rot1[1], rot1[2], rot1[3])
        elif len(rot1) == 5:
            seq.add_block(rot1[0], rot1[1], rot1[2], rot1[3], rot1[4])
        else:
            raise TypeError("number of rotated inputs not supported")

        seq.add_block(pp.make_delay(delay_TE[j]))

        if i > 0:
            rot2 = rotate('z', phi, gx, adc)
            if len(rot2) == 1:
                seq.add_block(rot2[0])
            elif len(rot2) == 2:
                seq.add_block(rot2[0], rot2[1])
            elif len(rot2) == 3:
                seq.add_block(rot2[0], rot2[1], rot2[2])
            elif len(rot2) == 4:
                seq.add_block(rot2[0], rot2[1], rot2[2], rot2[3])
            elif len(rot2) == 5:
                seq.add_block(rot2[0], rot2[1], rot2[2], rot2[3], rot2[4])
            else:
                raise TypeError("number of rotated inputs not supported")


        else:
            rot3 = rotate('z', phi, gx)
            if len(rot3) == 1:
                seq.add_block(rot3[0])
            elif len(rot3) == 2:
                seq.add_block(rot3[0], rot3[1])
            elif len(rot3) == 3:
                seq.add_block(rot3[0], rot3[1], rot3[2])
            elif len(rot3) == 4:
                seq.add_block(rot3[0], rot3[1], rot3[2], rot3[3])
            elif len(rot3) == 5:
                seq.add_block(rot3[0], rot3[1], rot3[2], rot3[3], rot3[4])
            else:
                raise TypeError("number of rotated inputs not supported")

        rot4 = rotate('z', phi, gx_spoil, gz_spoil, pp.make_delay(delay_TR[j]))
        if len(rot4) == 1:
            seq.add_block(rot4[0])
        elif len(rot4) == 2:
            seq.add_block(rot4[0], rot4[1])
        elif len(rot4) == 3:
            seq.add_block(rot4[0], rot4[1], rot4[2])
        elif len(rot4) == 4:
            seq.add_block(rot4[0], rot4[1], rot4[2], rot4[3])
        elif len(rot4) == 5:
            seq.add_block(rot4[0], rot4[1], rot4[2], rot4[3], rot4[4])
        else:
            raise TypeError("number of rotated inputs not supported")


seq.plot()
plt.show()
# Prepare the sequence output for the scanner
seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'gre_rad')
seq.set_definition('TE', TE)
seq.set_definition('TR', TR)
seq.set_definition('Flipangle', alpha)
seq.set_definition('Nx', Nx)
seq.set_definition('Nr', Nr)
seq.write('gre_rad_pypulseq.seq')

# Trajectory calculation
k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()
plt.figure()
plt.plot(np.transpose(k_traj))
plt.figure()
plt.plot(k_traj[0], k_traj[1], 'b')
plt.plot(k_traj_adc[0], k_traj_adc[1], 'r.')
plt.show()


# Change the k-space-trajectory in a (256,128,2) format instead of (3, 32768) (for example)
# Useful for image reconstruction
k_traj_neu = np.zeros((Nx, Nr, 2))

k_traj_neu[:, :, 0] = np.reshape(k_traj_adc[0], (Nx, Nr), order='F')
k_traj_neu[:, :, 1] = np.reshape(k_traj_adc[1], (Nx, Nr), order='F')

# redefine limits so that the trajectory is between -0.5 and 0.5
max0 = round(np.max(k_traj_neu[:, :, 0]))
k_traj_neu[:, :, 0] = k_traj_neu[:, :, 0] / max0 * 0.5
k_traj_neu[:, :, 1] = k_traj_neu[:, :, 1] / max0 * 0.5

k_traj_neu = k_traj_neu.astype('float32')
