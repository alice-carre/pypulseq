import numpy as np
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
'''
Variable density Cartesian sampling taken from
https://github.com/js3611/Deep-MRI-Reconstruction/blob/master/utils/compressed_sensing.py
'''
def normal_pdf(length, sensitivity):
    return np.exp(-sensitivity * (np.arange(length) - length / 2)**2)

def cartesian_mask(shape, acc, sample_n=10):
    """
    Sampling density estimated from implementation of kt FOCUSS
    shape: tuple - of form (..., nx, ny)
    acc: float - doesn't have to be integer 4, 8, etc..
    """
    N, Nx, Ny = int(np.prod(shape[:-2])), shape[-2], shape[-1]
    pdf_x = normal_pdf(Nx, 0.5/(Nx/10.)**2)
    lmda = Nx/(2.*acc)
    n_lines = int(Nx / acc)
    # add uniform distribution
    pdf_x += lmda * 1./Nx
    if sample_n:
        pdf_x[Nx//2-sample_n//2:Nx//2+sample_n//2] = 0
        pdf_x /= np.sum(pdf_x)
        n_lines -= sample_n
    mask = np.zeros((N, Nx))
    for i in range(N):
        idx = np.random.choice(Nx, n_lines, False, pdf_x)
        mask[i, idx] = 1
    if sample_n:
        mask[:, Nx//2-sample_n//2:Nx//2+sample_n//2] = 1
    size = mask.itemsize
    mask = as_strided(mask, (N, Nx, Ny), (size * Nx, size, 0))
    mask = mask.reshape(shape)
    return mask
"""
msk = cartesian_mask([1, 256, 1], 4, sample_n=10)
#msk = cartesian_mask([50, 200, 1], 4, sample_n=10) #sample_n (le nombre d'echantillon au centre toujours égaux à 1)
#donne 50 muster de taille 200
#200/4 nombre de points égal à 1

fig, ax = plt.subplots(1,2)
ax[0].imshow(msk[:,:,0])
ax[0].set_xlabel('$k_y$')
ax[0].set_ylabel('Time / Images')
ax[1].plot(np.sum(msk[:,:,0], axis=0))
"""