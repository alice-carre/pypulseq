import math
import numpy as np
import copy
from types import SimpleNamespace

from pypulseq.add_gradients import add_gradients
from pypulseq.block_to_events import block_to_events


def get_grad_abs_mag(grad):
    if grad.type == 'trap':
        return abs(grad.amplitude)
    else:
        return max(abs(grad.waveform))


def scale_grad(grad, scale):
    if grad.type =='trap':
        grad.amplitude = grad.amplitude * scale
        grad.area = grad.area * scale
        grad.flat_area = grad.flat_area * scale
    else:
        grad.waveform = grad.waveform * scale
        grad.first = grad.first * scale
        grad.last = grad.last * scale
    return grad

def rotate(axis, angle, *args: SimpleNamespace):
    """ align set alignment of the objects in the block
       [...] = rotate(axis, angle, obj <, obj> ...);

       Rotates the corresponding gradient object(s) about the given axis by
       the specified amount. Gradients parallel to the rotation axis and
       non-gradient objects are not affected.
       Possible rotation axes are 'x', 'y' or 'z'.

       Returns either a cell-array of objects if one return parameter is
       provided or an explicit list of objects if multiple parameters are
       given. Can be used directly as a parameter of seq.addBlock()."""

    axes = np.array(['x', 'y', 'z'])
    # cycle through the objects and rotate gradients non-parallel to the given rotation axis
    # Rotated gradients assigned to the same axis are then added together
    # First create indexes of the objects to be bypassed or rotated
    irotate1 = []
    irotate2 = []
    ibypass = []
    axes2rot = axes[axes != axis]

    if len(axes2rot) != 2:
        raise RuntimeError(f'incorrect axis specification')
    events = args
    index = 0
    for event in events:
        if not isinstance(event, (dict, SimpleNamespace)):
            raise TypeError("input(s) should be of type SimpleNamespace or a dict() in case of LABELINC or LABELSET")
        if ((event.type != 'grad' and event.type != 'trap') or event.channel == axis):
            ibypass.append(index)
        else:
            if event.channel == axes2rot[0]:
                irotate1.append(index)
            else:
                if event.channel == axes2rot[1]:
                    irotate2.append(index)
                else:
                    ibypass.append(index)
        index += 1

    # now every gradient to be rotated generates two new gradients, one on the
    # original axis and one on the other from the axes2rot list

    max_mag = 0  # measure of the relevant amplitude
    rotated1 = [None]*(len(irotate1) + len(irotate2))
    rotated2 = [None]*(len(irotate1) + len(irotate2))

    for i in range(0, len(irotate1)):
        g = copy.deepcopy(events[irotate1[i]])
        g1 = copy.deepcopy(events[irotate1[i]])
        max_mag = max(max_mag, get_grad_abs_mag(g))
        rotated1[i] = copy.deepcopy(scale_grad(g, math.cos(angle)))
        g1 = scale_grad(g1, math.sin(angle))
        g1.channel = axes2rot[1]
        rotated2[i] = copy.deepcopy(g1)

    o = len(irotate1)

    for i in range(0, len(irotate2)):
        g = copy.deepcopy(events[irotate2[i]])
        g2 = copy.deepcopy(events[irotate2[i]])
        max_mag = max(max_mag, get_grad_abs_mag(g))
        rotated2[i+o] = copy.deepcopy(scale_grad(g, math.cos(angle)))
        g = scale_grad(g2, -math.sin(angle))
        g2.channel = axes2rot[0]
        rotated1[i+o] = copy.deepcopy(g2)
    thresh = 1e-6*max_mag

    #eliminate gradient components under a certain thresh value

    for i in range(len(rotated1)-1, -1, -1):
        if get_grad_abs_mag(rotated1[i]) < thresh:
            del rotated1[i]
    for i in range (len(rotated2)-1, -1, -1):
        if get_grad_abs_mag(rotated2[i]) < thresh:
            del rotated2[i]

    g_modif = [None] * 2

    if len(rotated1) > 1:
        g_modif[0] = add_gradients(rotated1)
    elif len(rotated1) != 0:
        g_modif[0] = rotated1[0]

    if len(rotated2) > 1:
        g_modif[1] = add_gradients(rotated2)
    elif len(rotated2) != 0:
        g_modif[1] = rotated2[0]
    for i in range (len(g_modif)-1, -1, -1):
        if g_modif[i] is None or get_grad_abs_mag(g_modif[i]) < thresh:
           del g_modif[i]

    #export
    bypass=[]
    for i in ibypass:
        bypass.append(events[i])
    out = bypass+g_modif
    return out


