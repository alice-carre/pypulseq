# spiral sequence
import math
import copy
import numpy as np
from matplotlib import pyplot as plt

import pypulseq as pp
from pypulseq.rotate import rotate
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.write_seq_definitions import write_seq_definitions

seq = pp.Sequence()  # Create a new sequence object
fov = 256e-3  # Define FOV
Nx = 256 # Define resolution
Ny = Nx
slice_thickness = 3e-3  # Slice thickness
N_slices = 1
phi = np.pi / 2
N_shot = 2
delta = 2 * np.pi/N_shot

# Golden Angle Case:
#delta = 2*np.pi - (2*np.pi) * (2/(1+np.sqrt(5))) # angular increment  # angle = 137.51°


# Set the system limits
system = pp.Opts(max_grad=30, grad_unit='mT/m', max_slew=128, slew_unit='T/m/s', rf_ringdown_time=30e-6,
                 rf_dead_time=100e-6, adc_dead_time=10e-6)

# Create 90 degree slice selection pulse and gradient
rf, gz, gzr = pp.make_sinc_pulse(flip_angle=phi, duration=3e-3, slice_thickness=slice_thickness,
                                 apodization=0.5, time_bw_product=4, system=system, return_gz=True)

# Define k-space parameters
delta_k = 1 / fov
k_max = Nx / (2 * fov)  # kspace from -k_max to k_max
k_fov = 2 * k_max
delta_x = 1 / k_fov

# Variables given in Bernstein - chapter 17.6
# Warning: considered with the conversion from max_slew in T/m/s and max_grad in mT/m see convert function
lbd = N_shot / (2 * np.pi * fov)  # lambda
beta = system.max_slew / lbd  # beta = gamma * SR0 /lambda /2pi and system.max_slew = gamma * SR0/2pi
capital_lbd = 1  # >=1 to adjust the slew rate at t=0
teta_max = k_max / lbd

a2 = (9 * beta / 4) ** (1 / 3)
Ts = (3 / 2 * system.max_grad / (lbd * a2 ** 2)) ** 3
teta_s = 1 / 2 * beta * Ts ** 2 / (capital_lbd + beta / 2 / a2 * Ts ** (4 / 3))
T_acq = Ts + lbd / (system.max_grad * 2) * (teta_max ** 2 - teta_s ** 2)
if Ts>T_acq:
    T_acq = 2*np.pi*fov/(3*N_shot)*np.sqrt(1/(2*system.max_slew*(delta_x)**3)) # for large Go, slow SR0, small Nshot, or large FOV


# Definition of a desired number of samples - copied on the matlab code # not sure if appropriate
k_samples = round(T_acq/system.grad_raster_time)

t = np.linspace(0, T_acq, k_samples)

ka = np.zeros((2, k_samples))
for c in range(k_samples):
    if t[c] < Ts:
        teta = 1 / 2 * beta * t[c] ** 2 / (capital_lbd + beta / 2 / a2 * t[c] ** (4 / 3))
    else:
        teta = np.sqrt(teta_s ** 2 + 2 * system.max_grad / lbd * (t[c] - Ts))
    ka[0][c] = np.real(lbd * teta * np.exp(1j * (teta)))
    ka[1][c] = np.imag(lbd * teta * np.exp(1j * (teta)))

    # Calculate gradients and slew rates
ga, sa = pp.traj_to_grad(ka)

# Create 90 degree slice selection pulse and gradient
rf, gz, gzr = pp.make_sinc_pulse(flip_angle=np.pi / 2, duration=3e-3, slice_thickness=slice_thickness,
                                 apodization=0.5, time_bw_product=4, system=system, return_gz=True)
gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, system=system)

spiral_grad_shape = ga
spiral_grad_shape = np.c_[spiral_grad_shape, spiral_grad_shape[:, -1]]

gx = make_arbitrary_grad(channel='x', waveform=spiral_grad_shape[0], delay=pp.calc_duration(gz_reph))
gy = make_arbitrary_grad(channel='y', waveform=spiral_grad_shape[1], delay=pp.calc_duration(gz_reph))

# Spoilers
gz_spoil = pp.make_trapezoid(channel='z', area=4 * Nx * delta_k, system=system)
gx_spoil = pp.make_extended_trapezoid(channel='x', amplitudes=[spiral_grad_shape[0][-1], 0],
                                      times=[0, pp.calc_duration(gz_spoil)])
gy_spoil = pp.make_extended_trapezoid(channel='y', amplitudes=[spiral_grad_shape[1][-1], 0],
                                      times=[0, pp.calc_duration(gz_spoil)])

# Calculate ADC
#To be tested on the scanner
adc_time = T_acq
adc_samples = round(k_samples)
adc_dwell = round(adc_time / adc_samples / 100e-9) * 100e-9  # on Siemens adc_dwell needs to be aligned to 100ns
adc = pp.make_adc(num_samples=adc_samples, dwell=adc_dwell, delay=pp.calc_duration(gz_reph))

"""
# Calculate ADC
adc_time = np.shape(spiral_grad_shape)[1]*system.grad_raster_time
# the (Siemens) interpreter sequence
# per default will try to split the trajectory into segments <=1000 samples
# and every of these segments will have to have duration aligned to the
# gradient raster time
adc_samples_per_segment = 1000  # may be needed to play with this number to fill the entire trajectory
adc_samples_desired = k_samples
adc_segments = round(adc_samples_desired / adc_samples_per_segment)
adc_samples = adc_segments * adc_samples_per_segment
adc_dwell = round(adc_time / adc_samples / 100e-9) * 100e-9  # on Siemens adc_dwell needs to be aligned to 100ns

adc_segment_duration = adc_samples_per_segment * adc_dwell
if np.floor(divmod(adc_segment_duration, system.grad_raster_time)[1]) > np.finfo(float).eps:
    raise TypeError("ADC segmentation model results in incorrect segment duration")
# Update segment count
adc_segments = np.floor(adc_time / adc_segment_duration)
adc_samples = adc_segments * adc_samples_per_segment
adc = pp.make_adc(num_samples=adc_samples, dwell=adc_dwell, delay=pp.calc_duration(gz_reph))
"""

# Define sequence blocks
for s in range(0, N_slices):
    for i in range(0, N_shot):
        # seq.add_block(rf_fs,gz_fs) # fat-sat
        rf.freq_offset = gz.amplitude * slice_thickness * (s - (N_slices - 1) / 2)
        seq.add_block(rf, gz)

        # Rotation of gradient object(s) about the given axis and projection on cartesian axis to add it as a block to the sequence
        # Not very clean way to add events but the best way I found since add_block wouldn't take a list[SimpleNameSpace] as argument
        # The list returned by the rotate function has a variable length since sometimes gradients have to be projected on one or two axis

        rot1 = rotate('z', i*delta, gz_reph, gx, gy, adc)
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

        rot2 = rotate('z', i*delta, gx_spoil, gy_spoil, gz_spoil)
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

# Check whether the timing of the sequence is correct
ok, error_report = seq.check_timing()
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows\n')
    print(error_report)

write_seq_definitions(seq, fov=fov, slice_thickness=slice_thickness, Name='spiral', alpha = phi, Nx=Nx,
                      Sampling_scheme='spiral', Ny=Ny, N_slices=N_slices, N_interleaves = N_shot)
#seq.set_definition('MaxAdcSegmentLength', adc_samples_per_segment)
seq.write('spiral.seq')  # Output sequence for scanner

plt.figure()
seq.plot()
# Single-function for trajectory calculation
k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

# Plot k-spaces
plt.figure()
plt.plot(np.transpose(k_traj))
plt.figure()
plt.plot(k_traj[0], k_traj[1], 'b')
plt.plot(k_traj_adc[0], k_traj_adc[1], 'r.')
plt.show()

