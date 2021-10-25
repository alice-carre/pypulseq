# spiral sequence
import math
import copy
import numpy as np
from matplotlib import pyplot as plt


import pypulseq as pp
from pypulseq.rotate import rotate
from pypulseq.make_arbitrary_grad import make_arbitrary_grad

seq = pp.Sequence() #Create a new sequence object
fov = 256e-3 #Define FOV
Nx = 96 #Define resolution
Ny = Nx
slice_thickness = 3e-3  # slice thickness
N_slices = 1
oversampling = 2
phi = np.pi/2

#Set the system limits
system = pp.Opts(max_grad=30, grad_unit='mT/m', max_slew=120, slew_unit='T/m/s', rf_ringdown_time=30e-6,
                 rf_dead_time=100e-6, adc_dead_time=10e-6)
#Create fat-sat pulse
#(in Siemens interpreter from January 2019 duration is limited to 8.192 ms, and although product EPI uses 10.24 ms,
# 8 ms seems to be sufficient)
B0 = 2.89
sat_ppm = -3.45
sat_freq = sat_ppm * 1e-6 * B0 * system.gamma
rf_fs = pp.make_gauss_pulse(flip_angle=110*np.pi/180, bandwidth=np.abs(sat_freq), duration=8e-3, freq_offset=sat_freq,
                            system=system)
#spoil up to 0.1mm
gz_fs = pp.make_trapezoid(channel='z', area=1/1e-4, delay=pp.calc_duration(rf_fs), system=system)

#Create 90 degree slice selection pulse and gradient
rf, gz, gzr = pp.make_sinc_pulse(flip_angle=np.pi/2, duration=3e-3, slice_thickness=slice_thickness,
                                 apodization=0.5, time_bw_product=4, system=system, return_gz=True)
#Define k-space parameters
delta_k = 1 / fov
k_radius = int(np.round(Nx/2))
k_samples = int(np.round(2*np.pi*k_radius)*oversampling)
readout_time = 4.2e-4

#calculate a raw Archimedian spiral trajectory
ka = []
ka=np.ones((2, k_radius*k_samples+1))*1j
for c in range(0, k_radius*k_samples+1):
    r = delta_k*c/k_samples
    a = divmod(c, k_samples)[1] * 2 * np.pi/k_samples
    ka[0][c] = np.real(r * np.exp(1j*a))
    ka[1][c] = np.imag(r * np.exp(1j*a))

#ka = [np.real(ka), np.real(ka)]
# calculate gradients and slew rates
ga, sa = pp.traj_to_grad(ka)

#Limit analysis
safety_margin = 0.94 #needed to avoid violate the slew rate due to the rounding errors
dt_g_comp = abs(ga)/(system.max_grad*safety_margin)*system.grad_raster_time
dt_g_abs = abs(ga[0, :]+1j * ga[1, :]) / (system.max_grad*safety_margin) * system.grad_raster_time
#dt_s_comp
dt_s_abs = np.zeros(len(sa[0]))
dt_s_comp = np.zeros((len(sa), len(sa[0])))
for i in range(len(sa[0])):
    dt_s_abs[i] = (math.sqrt(abs(sa[0][i] + 1j * sa[1][i]) / (system.max_slew * safety_margin)) * system.grad_raster_time)
    for j in range(len(sa)):
        dt_s_comp[j][i] = math.sqrt(abs(sa[j][i])/(system.max_slew*safety_margin))*system.grad_raster_time
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

#Apply the lower limit not to lose the trajectory detail
dt_min = 4 * system.grad_raster_time/k_samples #we want at least 4 points per revolution
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
t_smooth = np.zeros(len(dt_smooth)+1)
t_smooth[1:] = np.cumsum(dt_smooth)
t_rough = np.zeros(len(dt_rough)+1)
t_rough[1:] = np.cumsum(dt_rough)

interp1 = [float(i)*system.grad_raster_time for i in range(0, 1+int(np.floor(t_smooth[-1]/system.grad_raster_time)))]
kopt_smooth = np.zeros((2,len(interp1)))
kopt_smooth[0] = np.interp(interp1, t_smooth, ka[0])
kopt_smooth[1] = np.interp(interp1, t_smooth, ka[1])
interp2 =[float(i)*system.grad_raster_time for i in range(0, 1+int(np.floor(t_rough[-1]/system.grad_raster_time)))]
kopt_rough = np.zeros((2,len(interp2)))
kopt_rough[0] = np.interp(interp2, t_rough, ka[0])
kopt_rough[1] = np.interp(interp2, t_rough, ka[1])

#analysis
print("duration orig %2d us\n" %np.round(1e6*system.grad_raster_time*len(ka[0])))
print("duration smooth %2d us\n" %np.round(1e6*system.grad_raster_time*len(kopt_smooth[0])))
print("duration rough %2d us\n" %np.round(1e6*system.grad_raster_time*len(kopt_rough[0])))

gos, sos = pp.traj_to_grad(kopt_smooth)
gor, sor = pp.traj_to_grad(kopt_rough)
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

#Define gradients and ADC events
spiral_grad_shape = gos
#Create 90 degree slice selection pulse and gradient
rf, gz, gzr = pp.make_sinc_pulse(flip_angle=np.pi / 2, duration=3e-3, slice_thickness=slice_thickness,
                                 apodization=0.5, time_bw_product=4, system=system, return_gz=True)
gz_reph = pp.make_trapezoid(channel='z', area=-gz.area / 2, system=system)

#Calculate ADC
#round down dwell time to 10 ns
adc_time = system.grad_raster_time*len(spiral_grad_shape[0])
# actually it is trickier than that: the (Siemens) interpreter sequence
# per default will try to split the trajectory into segments <=1000 samples
# and every of these segments will have to have duration aligned to the
# gradient raster time
adc_samples_per_segment = 1000 #may be needed to play with this number to fill the entire trajectory
adc_samples_desired = k_radius*k_samples
adc_segments = round(adc_samples_desired/adc_samples_per_segment)
adc_samples = adc_segments*adc_samples_per_segment
adc_dwell = round(adc_time/adc_samples/100e-9)*100e-9 # on Siemens adcDwell needs to be aligned to 100ns
adc_segment_duration = adc_samples_per_segment*adc_dwell
if np.floor(divmod(adc_segment_duration, system.grad_raster_time)[1]) > np.finfo(float).eps:
    raise TypeError("ADC segmentation model results in incorrect segment duration")
#update segment count
adc_segments = np.floor(adc_time/adc_segment_duration)
adc_samples = adc_segments * adc_samples_per_segment
adc = pp.make_adc(num_samples=adc_samples, dwell=adc_dwell, delay=pp.calc_duration(gz_reph))

#extend spiral_grad_shape by repeating the last sample
#this is needed to accomodate for the ADC tuning delay
spiral_grad_shape = np.c_[spiral_grad_shape, spiral_grad_shape[:, -1]]

#readout grad
gx = make_arbitrary_grad(channel='x', waveform=spiral_grad_shape[0], delay=pp.calc_duration(gz_reph))
gy = make_arbitrary_grad(channel='y', waveform=spiral_grad_shape[1], delay=pp.calc_duration(gz_reph))

#spoilers
gz_spoil = pp.make_trapezoid(channel='z', area=4 * Nx * delta_k, system=system)
gx_spoil = pp.make_extended_trapezoid(channel='x', amplitudes=[spiral_grad_shape[0][-1], 0], times=[0, pp.calc_duration(gz_spoil)])
gy_spoil = pp.make_extended_trapezoid(channel='y', amplitudes=[spiral_grad_shape[1][-1], 0], times=[0, pp.calc_duration(gz_spoil)])

#Define sequence blocks
for s in range(0, N_slices):
    seq.add_block(rf_fs,gz_fs) #fat-sat
    rf.freq_offset = gz.amplitude * slice_thickness * (s-(N_slices-1)/2)
    seq.add_block(rf, gz)

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

    rot2 = rotate('z', phi, gx_spoil,gy_spoil,gz_spoil)
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

seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'spiral')
seq.set_definition('MaxAdcSegmentLength', adc_samples_per_segment)
seq.write('spiral.seq') #output sequence for scanner

seq.plot()
#single-function for trajectory calculation
k_traj_adc, k_traj, t_excitation, t_refocusing, t_adc = seq.calculate_kspace()

#plot k-spaces
plt.figure()
plt.plot(np.transpose(k_traj))
plt.figure()
plt.plot(k_traj[0], k_traj[1], 'b')
plt.plot(k_traj_adc[0], k_traj_adc[1], 'r.')

