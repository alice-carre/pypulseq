# spiral sequence
import math
import copy
import numpy as np
from matplotlib import pyplot as plt

import pypulseq as pp
from pypulseq.rotate import rotate
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
#from pypulseq.write_seq_definitions import write_seq_definitions
from vds import vds


seq = pp.Sequence()  # Create a new sequence object
fov = 256e-3  # Define FOV
Nx = 256# Define resolution
Ny = Nx
slice_thickness = 3e-3  # Slice thickness
N_slices = 1
#oversampling = 2.5
phi = np.pi / 2
N_shot = 2 # Interleaves
delta = 2 * np.pi/N_shot

# Golden Angle Case:
#delta = 2*np.pi - (2*np.pi) * (2/(1+np.sqrt(5))) # angular increment # full spokes # angle = 137.51°

# Set the system limits
system = pp.Opts(max_grad=30, grad_unit='mT/m', max_slew=128, slew_unit='T/m/s', rf_ringdown_time=30e-6,
                 rf_dead_time=100e-6, adc_dead_time=10e-6)

# Create 90 degree slice selection pulse and gradient
rf, gz, gzr = pp.make_sinc_pulse(flip_angle=phi, duration=3e-3, slice_thickness=slice_thickness,
                                 apodization=0.5, time_bw_product=4, system=system, return_gz=True)

# Define k-space parameters
delta_k = 1 / fov
smax =system.max_slew #'Hz/m/s'
gmax = system.max_grad #'Hz/m'

T0 =system.grad_raster_time # Seconds
Fcoeff = [fov, -0] # FOV decreases linearly from Fcoeff[0] to Fcoeff[0]-Fcoeff[1]. #mm
#if Fcoeff = [fov, 0], fov=constante corresponds to the value given at the beginning of the program
#if Fcoeff = [0.240, -0.120] FOV decreases linearly from 24 cm to 12 cm

res = fov/Nx #m #resolution

rmax = 0.5/res #m^(-1)

k,g,s,time,r,theta = vds(smax, gmax, T0, N_shot, Fcoeff, rmax) # k-trajectory, g-gradient, s-slew, r-radius

# Create 90 degree slice selection pulse and gradient
rf, gz, gzr = pp.make_sinc_pulse(flip_angle=np.pi / 2, duration=3e-3, slice_thickness=slice_thickness,
                                 apodization=0.5, time_bw_product=4, system=system, return_gz=True)
gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, system=system)

# Calculate ADC
#To be tested on the scanner
adc_time = np.shape(g)[0]*T0
adc_samples = round(adc_time/T0)
adc_dwell = round(adc_time / adc_samples / 100e-9) * 100e-9  # on Siemens adc_dwell needs to be aligned to 100ns
#adc = pp.make_adc(num_samples=np.shape(k)[0], dwell=T0, delay=pp.calc_duration(gz_reph))
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
        spiral_grad_shape = np.zeros((2, np.shape(g)[0]))
        spiral_grad_shape[0] = np.real(g * np.exp(1j * delta * i))
        spiral_grad_shape[1] = np.imag(g * np.exp(1j * delta * i))
        # Extend spiral_grad_shape by repeating the last sample
        # This is needed to accomodate for the ADC tuning delay
        spiral_grad_shape = np.c_[spiral_grad_shape, spiral_grad_shape[:, -1]]

        # Readout grad
        #plt.plot(spiral_grad_shape[0])
        gx = make_arbitrary_grad(channel='x', waveform=spiral_grad_shape[0], delay=pp.calc_duration(gz_reph))
        gy = make_arbitrary_grad(channel='y', waveform=spiral_grad_shape[1], delay=pp.calc_duration(gz_reph))

        # Spoilers
        gz_spoil = pp.make_trapezoid(channel='z', area=4 * Nx * delta_k, system=system)
        gx_spoil = pp.make_extended_trapezoid(channel='x', amplitudes=[spiral_grad_shape[0][-1], 0],
                                              times=[0, pp.calc_duration(gz_spoil)])
        gy_spoil = pp.make_extended_trapezoid(channel='y', amplitudes=[spiral_grad_shape[1][-1], 0],
                                              times=[0, pp.calc_duration(gz_spoil)])
        rf.freq_offset = gz.amplitude * slice_thickness * (s - (N_slices - 1) / 2)
        seq.add_block(rf, gz)

        # Rotation of gradient object(s) about the given axis and projection on cartesian axis to add it as a block to the sequence
        # Not very clean way to add events but the best way I found since add_block wouldn't take a list[SimpleNameSpace] as argument
        # The list returned by the rotate function has a variable length since sometimes gradients have to be projected on one or two axis

        seq.add_block(gz_reph, gx, gy, adc)
        seq.add_block(gx_spoil, gy_spoil, gz_spoil)
# Check whether the timing of the sequence is correct
ok, error_report = seq.check_timing()
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows\n')
    print(error_report)

#write_seq_definitions(seq, fov=fov, slice_thickness=slice_thickness, Name='spiral', alpha = phi, Nx=Nx,
#                      Sampling_scheme='spiral', Ny=Ny, N_slices=N_slices, N_interleaves = N_shot)
seq.set_definition('delta', delta)
seq.write('spiral.seq')  # Output sequence for scanner

plt.figure()
#seq.plot()
# Single-function for trajectory calculation
k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

# Plot k-spaces
plt.figure()
plt.plot(np.transpose(k_traj))
plt.figure()
plt.plot(k_traj[0], k_traj[1], 'b')
plt.plot(k_traj_adc[0], k_traj_adc[1], 'r.')
plt.show()

