# spiral sequence #Matlab translation
import math
import copy
import numpy as np
from matplotlib import pyplot as plt

import pypulseq as pp
from pypulseq.rotate import rotate
from pypulseq.make_arbitrary_grad import make_arbitrary_grad

seq = pp.Sequence()  # Create a new sequence object
fov = 256e-3  # Define FOV
Nx = 192 # Define resolution
Ny = Nx
slice_thickness = 3e-3  # Slice thickness
N_slices = 1
oversampling = 1
phi = np.pi / 2
N_shot = 3

# Set the system limits
system = pp.Opts(max_grad=30, grad_unit='mT/m', max_slew=120, slew_unit='T/m/s', rf_ringdown_time=30e-6,
                 rf_dead_time=100e-6, adc_dead_time=10e-6)
"""
# Create fat-sat pulse
# (in Siemens interpreter from January 2019 duration is limited to 8.192 ms, and although product EPI uses 10.24 ms,
# 8 ms seems to be sufficient)
B0 = 2.89
sat_ppm = -3.45
sat_freq = sat_ppm * 1e-6 * B0 * system.gamma
rf_fs = pp.make_gauss_pulse(flip_angle=110 * np.pi / 180, bandwidth=np.abs(sat_freq), duration=8e-3,
                            freq_offset=sat_freq,
                            system=system)
# Spoil up to 0.1mm
gz_fs = pp.make_trapezoid(channel='z', area=1/1e-4, delay=pp.calc_duration(rf_fs), system=system)
"""
# Create 90 degree slice selection pulse and gradient
rf, gz, gzr = pp.make_sinc_pulse(flip_angle=np.pi / 2, duration=3e-3, slice_thickness=slice_thickness,
                                 apodization=0.5, time_bw_product=4, system=system, return_gz=True)
# Define k-space parameters
delta_k = 1 / fov
k_radius = int(np.round(Nx / 2))
k_samples = int(np.round(2 * np.pi * k_radius) * oversampling)
#k_samples = int(np.round(np.pi * k_radius) * oversampling)
List_grad_and_slew_rate_original = []
List_grad_and_slew_rate_rough = []
List_grad_and_slew_rate_smooth = []
for i in range(N_shot):
    # Calculate a raw Archimedian spiral trajectory
    ka = np.zeros((2, int(np.round((k_radius * k_samples + 1)/N_shot))))
    for c in range(0, int(np.round((k_radius * k_samples + 1)/N_shot))):
        r = delta_k * c / k_samples
        a = divmod(c, k_samples)[1] * 2 * np.pi / k_samples + 2 * np.pi * i / N_shot

        ka[0][c] = np.real(r * np.exp(1j * a))
        ka[1][c] = np.imag(r * np.exp(1j * a))
    #plt.plot(ka[0], ka[1])

    # Calculate gradients and slew rates
    ga, sa = pp.traj_to_grad(ka)
    List_grad_and_slew_rate_original.append((ga, sa))

    # Limit analysis
    safety_margin = 0.94  # Needed to avoid violate the slew rate due to the rounding errors
    dt_g_comp = abs(ga) / (system.max_grad * safety_margin) * system.grad_raster_time
    dt_g_abs = abs(ga[0, :] + 1j * ga[1, :]) / (system.max_grad * safety_margin) * system.grad_raster_time

    dt_s_abs = np.zeros(len(sa[0]))
    dt_s_comp = np.zeros((len(sa), len(sa[0])))

    for i in range(len(sa[0])):
        dt_s_abs[i] = (
                math.sqrt(abs(sa[0][i] + 1j * sa[1][i]) / (system.max_slew * safety_margin)) * system.grad_raster_time)
        for j in range(len(sa)):
            dt_s_comp[j][i] = math.sqrt(abs(sa[j][i]) / (system.max_slew * safety_margin)) * system.grad_raster_time
    """
    plt.figure()
    plt.plot(dt_g_comp.max(axis=0))
    plt.plot(dt_s_comp.max(axis=0))
    plt.plot(dt_g_abs)
    plt.plot(dt_s_abs)
    plt.title('time stepping defined by gradient and slew-rate')
    """

    dt_smooth = np.zeros(len(sa[0]))
    dt_rough = np.zeros(len(sa[0]))
    for i in range(len(sa[0])):
        dt_smooth[i] = max(dt_g_abs[i], dt_s_abs[i])
        dt_rough[i] = max(dt_g_comp.max(axis=0)[i], dt_s_comp.max(axis=0)[i])

    # Apply the lower limit not to lose the trajectory detail
    dt_min = 4 * system.grad_raster_time / k_samples  # We want at least 4 points per revolution
    dt_smooth0 = dt_smooth
    dt_rough0 = dt_rough
    dt_smooth[dt_smooth < dt_min] = dt_min
    dt_rough[dt_rough < dt_min] = dt_min
    """
    plt.figure()
    plt.plot(dt_smooth0)
    plt.plot(dt_smooth)
    plt.plot(dt_rough0)
    plt.plot(dt_rough)
    plt.title('combined time stepping')
    """

    t_smooth = np.zeros(len(dt_smooth) + 1)
    t_smooth[1:] = np.cumsum(dt_smooth)
    t_rough = np.zeros(len(dt_rough) + 1)
    t_rough[1:] = np.cumsum(dt_rough)

    interp1 = [float(i) * system.grad_raster_time for i in
               range(0, 1 + int(np.floor(t_smooth[-1] / system.grad_raster_time)))]
    k_opt_smooth = np.zeros((2, len(interp1)))
    k_opt_smooth[0] = np.interp(interp1, t_smooth, ka[0])
    k_opt_smooth[1] = np.interp(interp1, t_smooth, ka[1])
    interp2 = [float(i) * system.grad_raster_time for i in
               range(0, 1 + int(np.floor(t_rough[-1] / system.grad_raster_time)))]
    k_opt_rough = np.zeros((2, len(interp2)))
    k_opt_rough[0] = np.interp(interp2, t_rough, ka[0])
    k_opt_rough[1] = np.interp(interp2, t_rough, ka[1])

    # Analysis
    # print("duration orig %2d us\n" % np.round(1e6 * system.grad_raster_time * len(ka[0])))
    # print("duration smooth %2d us\n" % np.round(1e6 * system.grad_raster_time * len(k_opt_smooth[0])))
    # print("duration rough %2d us\n" % np.round(1e6 * system.grad_raster_time * len(k_opt_rough[0])))

    gos, sos = pp.traj_to_grad(k_opt_smooth)
    gor, sor = pp.traj_to_grad(k_opt_rough)

    List_grad_and_slew_rate_smooth.append((gos, sos))
    List_grad_and_slew_rate_rough.append((gor, sor))

"""

    plt.figure()
    plt.plot(gos[0])
    plt.plot(gos[1])
    plt.plot(abs(gos[0, :]+1j*gos[1, :]))
    plt.title('gradient with smooth (abs) constraint')

    plt.figure()
    plt.plot(gor[0])
    plt.plot(gor[1])
    plt.plot(abs(gor[0, :]+1j*gor[1, :]))
    plt.title('gradient with rough (component) constraint')

    plt.figure()
    plt.plot(sos[0])
    plt.plot(sos[1])
    plt.plot(abs(sos[0, :]+1j*sos[1, :]))
    plt.title('slew rate with smooth (abs) constraint')

    plt.figure()
    plt.plot(sor[0])
    plt.plot(sor[1])

    plt.plot(abs(sor[0, :]+1j*sor[1, :]))
    plt.title('slew rate with rough (component) constraint')
"""

# Define gradients and ADC events

# Create 90 degree slice selection pulse and gradient
rf, gz, gzr = pp.make_sinc_pulse(flip_angle=np.pi / 2, duration=3e-3, slice_thickness=slice_thickness,
                                 apodization=0.5, time_bw_product=4, system=system, return_gz=True)
gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, system=system)
List_spiral_grad_shape = []
len_to_save = 0
List_gradients = []
for i in range(0, N_shot):
    List_spiral_grad_shape.append(List_grad_and_slew_rate_smooth[i][0])
    spiral_grad_shape = List_spiral_grad_shape[i]
    len_to_save = len(spiral_grad_shape[0])
    # Extend spiral_grad_shape by repeating the last sample
    # This is needed to accomodate for the ADC tuning delay
    spiral_grad_shape = np.c_[spiral_grad_shape, spiral_grad_shape[:, -1]]

    # Readout grad
    gx = make_arbitrary_grad(channel='x', waveform=spiral_grad_shape[0], delay=pp.calc_duration(gz_reph))
    gy = make_arbitrary_grad(channel='y', waveform=spiral_grad_shape[1], delay=pp.calc_duration(gz_reph))

    # Spoilers
    gz_spoil = pp.make_trapezoid(channel='z', area=4 * Nx * delta_k, system=system)
    gx_spoil = pp.make_extended_trapezoid(channel='x', amplitudes=[spiral_grad_shape[0][-1], 0],
                                          times=[0, pp.calc_duration(gz_spoil)])
    gy_spoil = pp.make_extended_trapezoid(channel='y', amplitudes=[spiral_grad_shape[1][-1], 0],
                                          times=[0, pp.calc_duration(gz_spoil)])
    List_gradients.append((gx, gy, gx_spoil, gy_spoil))

# Calculate ADC
# Round down dwell time to 10 ns
adc_time = system.grad_raster_time * len_to_save
# the (Siemens) interpreter sequence
# per default will try to split the trajectory into segments <=1000 samples
# and every of these segments will have to have duration aligned to the
# gradient raster time
adc_samples_per_segment = 1000  # may be needed to play with this number to fill the entire trajectory
adc_samples_desired = k_radius * k_samples/N_shot  # adc_samples_desired = k_radius * k_samples//N_shot
adc_segments = round(adc_samples_desired / adc_samples_per_segment)
adc_samples = adc_segments * adc_samples_per_segment
adc_dwell = round(adc_time / adc_samples / 100e-9) * 100e-9  # on Siemens adc_dwell needs to be aligned to 100ns

# Ici test
#adc_dwell = (np.pi * k_radius ** 2 / adc_samples_desired) * system.grad_raster_time #L = 3,14 x R x (D + d) ÷ 2 avec R, bagues,D et d diametre interne externe
#ne marche pas avec le scanner pour 192 avec les 2/1 oversampling
adc_segment_duration = adc_samples_per_segment * adc_dwell
if np.floor(divmod(adc_segment_duration, system.grad_raster_time)[1]) > np.finfo(float).eps:
    raise TypeError("ADC segmentation model results in incorrect segment duration")
# Update segment count
adc_segments = np.floor(adc_time / adc_segment_duration)
adc_samples = adc_segments * adc_samples_per_segment
adc = pp.make_adc(num_samples=adc_samples, dwell=adc_dwell, delay=pp.calc_duration(gz_reph))

# Define sequence blocks
for s in range(0, N_slices):
    for i in range(0, N_shot):
        # seq.add_block(rf_fs,gz_fs) # fat-sat
        gx = List_gradients[i][0]
        gy = List_gradients[i][1]
        gx_spoil = List_gradients[i][2]
        gy_spoil = List_gradients[i][3]
        rf.freq_offset = gz.amplitude * slice_thickness * (s - (N_slices - 1) / 2)
        seq.add_block(rf, gz)

        # Rotation of gradient object(s) about the given axis and projection on cartesian axis to add it as a block to the sequence
        # Not very clean way to add events but the best way I found since add_block wouldn't take a list[SimpleNameSpace] as argument
        # The list returned by the rotate function has a variable length since sometimes gradients have to be projected on one or two axis

        rot1 = rotate('z', phi, gz_reph, gx, gy, adc)
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

        rot2 = rotate('z', phi, gx_spoil, gy_spoil, gz_spoil)
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

seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'spiral')
seq.set_definition('MaxAdcSegmentLength', adc_samples_per_segment)
seq.set_definition('Sampling_scheme', 'spiral')
seq.write('spiral.seq')  # Output sequence for scanner

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
