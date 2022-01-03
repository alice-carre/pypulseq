import numpy as np
import matplotlib.pyplot as plt


# load data from echo
praw = r'Z:/_allgemein/projects/Katja/forOthers/forChristophK/girf/'
# praw = ''

# load data
triangles_meas_load = -(np.load(praw + 'triangles_meas.npy'))   # load with get_meas_data(), -: match with input
triangles_meas = triangles_meas_load.swapaxes(2, 1)
input_triangle = np.load(praw + 'input_triangle.npy')
girf = np.load(praw + 'girf.npy')
freq_range_zp = np.load(praw + 'freq_range_zp.npy')



def gradient_distort_GIRF(grad_input, grt, h, f_range, nb_zeros):
    """
    The function gradient_distort_GIRF is performed using modified code of Shaihan Malik
    (https://github.com/mriphysics/reVERSE-GIRF). Please cite appropriately.
    :param grad_input: nominal Gradient played out by scanner
    :param grt: GradientRasterTime, 10 us (Siemens)
    :param h: gradient impulse response function
    :param f_range: freq_range girf
    :param nb_zeros: padding samples to add at start and end
    :return: grad_distorted, grad_distorted_zeros
    """

    nb_pad = np.zeros((nb_zeros, 3))

    ff_hz = f_range * 1e3  # [kHz * 1e3 = Hz]
    dwelltime = grt * 1e-6    # dwelltime grad_input GRT [10us], [* 1e-6 in s]

    # gradient on all axis
    grad_input_xyz = np.stack([grad_input for _ in range(3)], axis=-1)


    # add zeros on front and end
    grad_input_zeros = np.concatenate([nb_pad, grad_input_xyz, nb_pad], axis=0)
    nb_gradient = grad_input_zeros.shape[0]
    t_input = np.arange(0, nb_gradient) * dwelltime


    # construct Fourier matrix
    df_act = ff_hz[1] - ff_hz[0]
    # df_req = 1 / t_input[-1]      todo: check if frequencies are sufficient
        # if df_req < df_act:
            # new frequency range

    ff = ff_hz[np.newaxis, :]
    ff_ = ff_hz[:, np.newaxis]
    t_in = t_input[np.newaxis, :]
    t_in_ = t_input[:, np.newaxis]


    F = dwelltime * np.exp(2 * np.pi * 1j * np.matmul(ff_, t_in))
    Fj = df_act * np.exp(-2 * np.pi * 1j * np.matmul(t_in_, ff))        # inverse


    # apply H
    grad_distorted_zeros = np.real(Fj @ (h * (F @ grad_input_zeros)))
    # remove zeros
    grad_distorted = grad_distorted_zeros[nb_zeros:(nb_gradient-nb_zeros)]

    return(grad_distorted, grad_distorted_zeros)



def plot_grad_predicted_measured(input, grad_meas, dt_meas, grad_distorted, dt_input, nb_triangle, axis, nb_zeros):
    nb = nb_triangle
    axis = axis
    axis_list = ['x-axis', 'y-axis', 'z-axis']

    # new try all: input, predicted, meas
    t_in = np.arange(0, len(input[2])) * dt_input * 1e-3  # [ms], [dt=10us]
    t_input = t_in - 50 * 1e-3   # [ms], -50 us before ADCstart at 0us
    t_meas = np.arange(0, grad_meas.shape[2]) * dt_meas * 1e-3      # [dt=8.7 us]
    t_grad_measured = t_meas - (50 - dt_meas) * 1e-3
    # -50us or -(50-8.7) or -(50-8.7/2) us, first sample at dwelltime or dwelltime/2)
    t_grad_distorted = np.arange(0, grad_distorted.shape[0]) * dt_input * 1e-3      # [ms], 10us GRT

    fig = plt.figure(figsize=(8, 3))
    # x-axis
    plt.plot(t_input, input[nb], label='nominal: 10 us')
    plt.plot(t_grad_measured, grad_meas[axis, nb, :], label='measured: 8.7 us')    # [axis, nb, :]
    plt.plot(t_grad_distorted - (50 - dt_meas) * 1e-3, grad_distorted[:, axis],
             label='predicted: 10 us')      # nb_zeros * 10 us: added zeros before and after gradient

    plt.title('Input and Output Gradients, ' + axis_list[axis])
    plt.legend()
    plt.xlabel('t $(ms)$')
    plt.ylabel(r'G ($\frac{mT}{m}$)')
    plt.tight_layout()

    # zoom in
    plt.figure(figsize=(3, 2))
    plt.plot(t_input, input[nb], label='nominal: 10 us')
    plt.plot(t_grad_measured, grad_meas[axis, nb, :], label='measured: 8.7 us')  # [axis, nb, :]
    plt.plot(t_grad_distorted - (50 - dt_meas) * 1e-3, grad_distorted[:, axis],
             label='predicted: 10 us')  # 20zeros=20*10us=200us
    plt.xlim(-0.03, 0.02)
    plt.ylim(-0.1, 2.5)
    plt.tight_layout()

    return()
"""
# predict input triangle with girf
grad_distorted, grad_distorted_zeros = gradient_distort_GIRF(grad_input=input_triangle[0], grt=10, h=girf,
                                                                        f_range=freq_range_zp, nb_zeros=20)

# plot predicted gradient and measures gradient
plot_grad_predicted_measured(input=input_triangle, grad_meas=triangles_meas, dt_meas=8.7,
                                        grad_distorted=grad_distorted, dt_input=10, nb_triangle=0, axis=0, nb_zeros=20)
"""
