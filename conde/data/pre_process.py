import numpy as np
import os
import segyio
import os
import h5py


def cut_seismic_slices(volume, size=128, inline_step=15, crossline_step=15):
    """
    Cuts a 3D seismic volume into 2D slices in both inline and crossline directions.

    Parameters:
    - volume (np.ndarray): seismic cube with shape [inline, crossline, depth]
    - size (int): patch size (height and width)
    - inline_step (int): step between extracted inlines
    - crossline_step (int): step between extracted crosslines

    Returns:
    - slices (list of np.ndarray): list of 2D patches, each of shape [size, size]
    """
    il, xl, z = volume.shape
    slices = []

    # INLINE slices: slicing over IL axis → shape [Z, XL]
    for i in range(0, il, inline_step):
        if z >= size and xl >= size:
            slice2d = volume[i, :, :]  # shape [XL, Z]
            for x in range(0, xl - size + 1, size):
                for d in range(0, z - size + 1, size):
                    patch = slice2d[x:x+size, d:d+size]
                    slices.append(patch)

    # CROSSLINE slices: slicing over XL axis → shape [IL, Z]
    for j in range(0, xl, crossline_step):
        if z >= size and il >= size:
            slice2d = volume[:, j, :]  # shape [IL, Z]
            for i in range(0, il - size + 1, size):
                for d in range(0, z - size + 1, size):
                    patch = slice2d[i:i+size, d:d+size]
                    slices.append(patch)

    return np.stack(slices)



def create_seismic_h5(h5_path, data_dict):
    """
    Creates an HDF5 file from multiple seismic slice sets.

    Parameters:
    - h5_path (str): output file path (e.g., 'seismics.h5')
    - data_dict (dict): { 'clean': [np.array], 'noise1': [np.array], ... }
                        All lists must have the same length.
    """
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    keys = list(data_dict.keys())
    num_slices = len(data_dict[keys[0]])
    shape_check = all(len(v) == num_slices for v in data_dict.values())
    if not shape_check:
        raise ValueError("All datasets must have the same number of slices.")

    with h5py.File(h5_path, "w") as f:
        for key, slices in data_dict.items():
            dset = f.create_dataset(
                key,
                data=np.stack(slices),
                dtype='float32',
                compression="gzip"
            )
    print(f"HDF5 file saved: {h5_path} with keys: {list(data_dict.keys())}")


if __name__ == '__main__':
    
    base_dir = '/pgeoprj/godeep/dados/l3_datasets/Modelos/Modelo3'
    files = ['Refletividade_Modelo3.segy', 
             'Sism14Hz_70perVel_Ruido0p5_0p5_Aten3_2___Modelo5_Transcorrente_HighAng_SHmax107_e0p012.segy',
            'Sism25Hz_30perVel_Ruido0p6_0p6_Aten10_Sal_Difra0p7_Modelo3_Transcorrente_HighAng_SHmax30_e0.segy',
            'Sism30Hz_70perVel_Ruido0p6_0p6_Aten11_1___Modelo5_Transcorrente_HighAng_SHmax107_e0p012.segy',
            'Sism35Hz_100perVel_Ruido0_0____Modelo3_Transcorrente_HighAng_SHmax30_e0.segy']

    data_dict = {}

    for file in files:
        file_path = f'{base_dir}/{file}'
        print(file_path)
        volume = segyio.tools.cube(file_path)
        slices = cut_seismic_slices(volume, size=224, inline_step=15, crossline_step=15)
        data_dict[file] = slices
        

    create_seismic_h5('/pgeoprj/godeep/ej49/ConDe/dataset/M3v2.h5', data_dict)






        