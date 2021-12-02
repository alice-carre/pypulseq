import pypulseq as pp
from pypulseq.Sequence.sequence import Sequence
"""
def write_seq_definitionss(seq: Sequence,
                   seq_defs: dict) \
        -> Sequence:

    Writes seq-file 'Definitions' from dictionary
    :param seq: pypulseq Sequence object
    :param seq_defs: dictionary with all entries that should be written into 'Definitions' of the seq-file
    :param use_matlab_names: set to True to use the same variable names as in Matlab
    :return:


    translator = {'number_of_readouts': 'kspace_encode_step_1',
                   'k_space_encoding1' : 'kspace_encode_step_2',
                     'N_slices' : 'slice',
                      : 'repetition',
                      : 'set',
                      : 'segment'
                      }\

        # create new dict with correct names (needs to be done before to be able to sort it correctly)
        dict_ = {}
        for k, v in seq_defs.items():
            # convert names
            if k in translator:
                k = translator[k]

            # write entry
            dict_.update({k: v})
    else:
        dict_ = seq_defs

    # write definitions in alphabetical order and convert to correct value types
    for k, v in sorted(dict_.items()):
        # convert value types
        if type(v) == np.ndarray:
            pass
        elif type(v) in [int, float, np.float32, np.float64, np.float]:
            v = str(round_number(v, 6))
        else:
            v = str(v)
        seq.set_definition(key=k, val=v)

    return seq


if hasattr(seq.dict_definitions, 'TI'):
    TI = seq.dict_definitions['TI']

if hasattr(seq.dict_definitions, 'offsets_ppm'):
    offsets_ppm = seq.dict_definitions['offsets_ppm']
    
    
# Set correct encoding and dims
mr.Pars.Encoding.Idx.Ki[1, :] = np.tile(idx_centric, n_rep)
mr.Pars.Encoding.Idx.Ki[8, :] = np.repeat(np.arange(n_rep), n_ph)

mr.Pars.Encoding.KDims[8] = n_rep
mr.Pars.Encoding.IDims = matrix[:3]
mr.Pars.Encoding.KFTDims = matrix[:3]

MR.Data.Nav.Idx.Ki[1, :] = h5_data['head']['idx']['kspace_encode_step_1'][nav_idx].astype(np.float32, casting='safe')  # Ky
        MR.Data.Nav.Idx.Ki[2, :] = h5_data['head']['idx']['kspace_encode_step_2'][nav_idx].astype(np.float32, casting='safe')  # Kz
        MR.Data.Nav.Idx.Ki[4, :] = h5_data['head']['idx']['slice'][nav_idx].astype(np.float32, casting='safe')  # Slices
        MR.Data.Nav.Idx.Ki[5, :] = h5_data['head']['idx']['average'][nav_idx].astype(np.float32, casting='safe')  # Averages
        MR.Data.Nav.Idx.Ki[6, :] = h5_data['head']['idx']['phase'][nav_idx].astype(np.float32, casting='safe')  # Phases
        MR.Data.Nav.Idx.Ki[7, :] = h5_data['head']['idx']['contrast'][nav_idx].astype(np.float32, casting='safe')  # Echoes
        MR.Data.Nav.Idx.Ki[8, :] = h5_data['head']['idx']['repetition'][nav_idx].astype(np.float32, casting='safe')  # Repetitions
        MR.Data.Nav.Idx.Ki[9, :] = h5_data['head']['idx']['set'][nav_idx].astype(np.float32, casting='safe')  # Sets
        MR.Data.Nav.Idx.Ki[10, :] = h5_data['head']['idx']['segment'][nav_idx].astype(np.float32, casting='safe')  # Segments

"""

def write_seq_definitions(seq: Sequence, fov: float, slice_thickness: float, Name: str, alpha: float, Nx: int, Sampling_scheme: str='cartesian',
                          Ny: int=1 , Nr: int=1, Nz: int = 1, N_slices: int=1, average: float=1, phase: float=1, contrast: float=1,
                          repetition: float = 1, set: float = 1, segment : float = 1, N_interleaves: float = 1, TE= 0, TR =0) -> None:

    seq.set_definition('FOV', [fov, fov, slice_thickness])
    #seq.set_definition('Name', 'gre_rad')
    if TE != 0:
        seq.set_definition('TE', TE)
    if TR != 0:
        seq.set_definition('TR', TR)

    seq.set_definition('Flipangle', alpha)
    seq.set_definition('Sampling_scheme', Sampling_scheme)

    if Sampling_scheme!= 'radial' and Sampling_scheme != 'cartesian' and Sampling_scheme != 'spiral':
        raise TypeError("Type of sampling scheme not supported")

    seq.set_definition('number_of_readouts', int(Nx)) #kx

    if Sampling_scheme == 'radial':
        seq.set_definition('number_of_spokes', int(Nr))
    else:
        seq.set_definition('k_space_encoding1', int(Ny))

    if Sampling_scheme == 'spiral':
        if N_interleaves !=1:
            seq.set_definition('N_interleaves', N_interleaves)

    if N_slices != 1:
        seq.set_definition('slices', N_slices )
    if average != 1:
        seq.set_definition('average', average)
    if phase !=1 :
        seq.set_definition('phase', phase )
    if contrast != 1:
        seq.set_definition('contrast', contrast )
    if repetition != 1:
        seq.set_definition('repetition', repetition)
    if set != 1:
        seq.set_definition('set', set)
    if segment != 1:
        seq.set_definition('segment', segment)
    if Nz != 1:
        seq.set_definition('k_space_encoding2', Nz)


def read_definitions(seq):
    # default values
    Sampling_scheme = 'cartesian'
    Ny = 1
    Nr = 1
    Nz = 1
    N_slices = 1
    average = 1
    phase = 1
    contrast = 1
    repetition = 1
    set = 1
    segment = 1
    N_interleaves = 1
    TE = 0
    TR = 0

    if 'FOV' in seq.dict_definitions:
        fov_x = seq.dict_definitions['FOV'][0]
        fov_y = seq.dict_definitions['FOV'][1]
        slice_thickness = seq.dict_definitions['FOV'][2]
    else:
        raise TypeError("FOV not given")

    if 'Sampling_scheme' in seq.dict_definitions:
        Sampling_scheme = seq.dict_definitions['Sampling_scheme'][0]

    if 'number_of_readouts' in seq.dict_definitions: #kx
        Nx = seq.dict_definitions['number_of_readouts'][0]
    else:
        raise TypeError("number_of_readouts not given")

    if 'k_space_encoding1' in seq.dict_definitions:
        Ny = seq.dict_definitions['k_space_encoding1'][0]

    if 'number_of_spokes' in seq.dict_definitions:
        Nr = seq.dict_definitions['number_of_spokes'][0]

    if 'slices' in seq.dict_definitions:
        N_slices = seq.dict_definitions['slices'][0]

    if 'k_space_encoding2' in seq.dict_definitions:
        Nz = seq.dict_definitions['k_space_encoding2'][0]

    if 'average' in seq.dict_definitions:
        average = seq.dict_definitions['average'][0]

    if 'phase' in seq.dict_definitions:
        phase = seq.dict_definitions['phase'][0]

    if 'contrast' in seq.dict_definitions:
        contrast = seq.dict_definitions['contrast'][0]

    if 'repetition' in seq.dict_definitions:
        repetition = seq.dict_definitions['repetition'][0]

    if 'set' in seq.dict_definitions:
        set = seq.dict_definitions['set'][0]

    if 'segment' in seq.dict_definitions:
        segment = seq.dict_definitions['segment'][0]

    if 'N_interleaves' in seq.dict_definitions:
        N_interleaves = seq.dict_definitions['N_interleaves'][0]

    dico = dict({'fov_x' : fov_x,
    'fov_y': fov_y,
    'slice_thickness' : slice_thickness,
    'Sampling_scheme' : Sampling_scheme,
    'number_of_readouts' : round(Nx),
    'k_space_encoding1' : round(Ny),
    'number_of_spokes': round(Nr),
    'slices' : round(N_slices),
    'average' : round(average),
    'phase' : round(phase),
    'contrast' : round(contrast),
    'repetition' : round(repetition),
    'set' : round(set),
    'segment' : round(segment),
    'N_interleaves' : round(N_interleaves),
    'k_space_encoding2' : round(Nz)})

    if dico.get('k_space_encoding1') == 1:
        dico['k_space_encoding1'] = dico['number_of_spokes']
    if dico.get('number_of_spokes') == 1:
        dico['number_of_spokes'] = dico['k_space_encoding1']

    if dico.get('k_space_encoding1') == 1 and dico.get('number_of_spokes') == 1:
        dico['k_space_encoding1'] = dico['number_of_readouts']

    return dico


