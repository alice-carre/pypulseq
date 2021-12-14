# spiral sequence
import math
import copy
import numpy as np
from matplotlib import pyplot as plt

import pypulseq as pp
from pypulseq.rotate import rotate
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.write_seq_definitions import write_seq_definitions

# Define parameters
fov = 256e-3  # Define FOV
Nx = Ny = 256  # Define resolution
slice_thickness = 3e-3  # Slice thickness
N_slices = 1
oversampling = 1
N_shot = 2 # only works for 1 or 2 interleaves so far
delta = 2 * np.pi/N_shot
phi=10 #grad
# Golden Angle Case:
#delta = 2*np.pi - (2*np.pi) * (2/(1+np.sqrt(5))) # angular increment # angle = 137.51°


# Set the system limits
seq = pp.Sequence()  # Create a new sequence object
system = pp.Opts(max_grad=20, grad_unit='mT/m', max_slew=100, slew_unit='T/m/s', rf_ringdown_time=30e-6,
                 rf_dead_time=100e-6, adc_dead_time=10e-6)

# Define k-space parameters
delta_k = 1 / fov
N_r = int(np.round(Nx / 2))
N_theta = int(np.round(2 * np.pi * N_r) * oversampling)
N_total = int(round(N_r * N_theta / N_shot)) # modification due to several interleaves : divide by N_shot

# Calculate a raw Archimedian spiral trajectory
ka = np.zeros((2, N_total))
for c in range(0, N_total):
    r = delta_k * c / N_theta * N_shot # modification due to several interleaves : multiply by N_shot
    a = divmod(c, N_theta)[1] * 2 * np.pi / N_theta
    ka[0][c] = np.real(r * np.exp(1j * a))  # convert to Cartesian coordinates
    ka[1][c] = np.imag(r * np.exp(1j * a))  # convert to Cartesian coordinates

# Calculate gradients and slew rates
ga, sa = pp.traj_to_grad(ka)

# calculate time stepping for gradient taking safety margin into account
safety_margin = 0.95
dt_g_abs = np.abs(ga[0, :] + 1j * ga[1, :]) / (system.max_grad * safety_margin) * system.grad_raster_time

# calculate time stepping for slew rate taking safety margin into account
dt_s_abs = np.zeros(sa.shape[1])
for i in range(sa.shape[1]):
    dt_s_abs[i] = np.sqrt(np.abs(sa[0][i] + 1j * sa[1][i]) / (system.max_slew * safety_margin)) \
                  * system.grad_raster_time

# select limiting time step from gradient and slew rate
dt_smooth = np.maximum(dt_g_abs, dt_s_abs)

# calculate timing
t_smooth = np.zeros(dt_smooth.size + 1)
t_smooth[1:] = np.cumsum(dt_smooth)

# calculate gradient timing
max_grad_values = int(np.ceil(t_smooth[-1] / system.grad_raster_time))
grad_timing = np.array([float(i) * system.grad_raster_time for i in range(0, max_grad_values)])

# calculate optimized trajectory
ka_opt = np.zeros((2, grad_timing.size))
ka_opt[0] = np.interp(grad_timing, t_smooth, ka[0])
ka_opt[1] = np.interp(grad_timing, t_smooth, ka[1])

# convert optimized trajectory to final gradient and slew rate values
ga_opt, sa_opt = pp.traj_to_grad(ka_opt)

# Create slice selection pulse and gradient (needs to be created before ADC to calculate correct ADC delay)
rf, gz, gzr = pp.make_sinc_pulse(flip_angle=phi*np.pi/180, duration=3e-3, slice_thickness=slice_thickness,
                                 apodization=0.5, time_bw_product=4, system=system, return_gz=True)
gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, system=system)

#To be tested on the scanner
adc_time = system.grad_raster_time * ga_opt.shape[1]
adc_samples = N_total
adc_dwell = round(adc_time / adc_samples / 100e-9) * 100e-9  # on Siemens adc_dwell needs to be aligned to 100ns
adc = pp.make_adc(num_samples=adc_samples, dwell=adc_dwell, delay=pp.calc_duration(gz_reph))

# Extend spiral_grad_shape by repeating the last sample
# This is needed to accomodate for the ADC tuning delay
spiral_grad_shape = np.c_[ga_opt, ga_opt[:, -1]]

# Create final gradients
gx = make_arbitrary_grad(channel='x', waveform=spiral_grad_shape[0], delay=pp.calc_duration(gz_reph))
gy = make_arbitrary_grad(channel='y', waveform=spiral_grad_shape[1], delay=pp.calc_duration(gz_reph))

# Spoilers
gz_spoil = pp.make_trapezoid(channel='z', area=4 * Nx * delta_k, system=system)
gx_spoil = pp.make_extended_trapezoid(channel='x', amplitudes=[spiral_grad_shape[0][-1], 0],
                                      times=[0, pp.calc_duration(gz_spoil)])
gy_spoil = pp.make_extended_trapezoid(channel='y', amplitudes=[spiral_grad_shape[1][-1], 0],
                                      times=[0, pp.calc_duration(gz_spoil)])

# Define sequence blocks

for s in range(0, N_slices):
    for index in range(0, N_shot):
        rf.freq_offset = gz.amplitude * slice_thickness * (s - (N_slices - 1) / 2)
        seq.add_block(rf, gz)

        # Rotation of gradient object(s) about the given axis and projection on cartesian axis to add it as a block to the sequence
        # Not very clean way to add events but the best way I found since add_block wouldn't take a list[SimpleNameSpace] as argument
        # The list returned by the rotate function has a variable length since sometimes gradients have to be projected on one or two axis

        rot1 = rotate('z', index * delta, gz_reph, gx, gy, adc)
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

        rot2 = rotate('z', index * delta, gx_spoil, gy_spoil, gz_spoil)
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

write_seq_definitions(seq, fov=fov, slice_thickness=slice_thickness, Name='spiral', alpha = np.pi/2, Nx=Nx,
                      Sampling_scheme='spiral', Ny=Ny, N_slices=N_slices, N_interleaves = N_shot)
#seq.set_definition('MaxAdcSegmentLength', adc_samples_per_segment)
seq.write('matlab_spiral.seq')  # Output sequence for scanner
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
