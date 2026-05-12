import numpy as np
import os
import pdb
import tensorflow as tf

from tqdm import tqdm

def np_to_tfdataset(arr1, arr2=None, batch_size=32, trim_batch=False):
    if arr2 is not None:
        assert arr1.shape == arr2.shape, f'Given arrays should have the same shape.'

    if arr2 is not None:
        data = tf.data.Dataset.from_tensor_slices((arr1, arr2))
    else:
        data = tf.data.Dataset.from_tensor_slices(arr1)

    data = data.batch(batch_size)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    
    data = data.with_options(options)

    return data

def load_dataset(data_path):
    files = sorted(os.listdir(data_path))
    files = [ os.path.join(data_path, f) for f in files if os.path.isfile(os.path.join(data_path, f)) and os.path.splitext(f)[1] == '.npy' ]
    
    pbar = tqdm(files, ncols=90)
    
    data = []
    for f in pbar: data.append( np.load(f) )
    
    data = np.vstack(data)
    #data = np.moveaxis(data, -1, 0)
    data = np.expand_dims(data, axis=-1)
    
    return data

def rescale_magnitude(data, t_min=0.0, t_max=1.0):
    r_min, r_max = np.min(data), np.max(data)
    num = data - r_min
    den = r_max - r_min

    return (num / den) * (t_max - t_min) + t_min

def pad_square(data, sag_axis, pad_to=None, pad_value=0.0):
    '''
    Given a 3D array, move given axis to last dimension and pad first two dimensions.
    '''
    data = np.moveaxis(data, sag_axis, -1)
    
    if pad_to is None: pad_to = max(data.shape[0], data.shape[1])

    pad_len_0 = (pad_to - data.shape[0]) // 2
    pad_len_1 = (pad_to - data.shape[1]) // 2
    data = np.pad(
            data, 
            [(pad_len_0, pad_len_0), (pad_len_1, pad_len_1), (0, 0)],
            constant_values=pad_value
        )
    
    data = np.moveaxis(data, -1, sag_axis)

    return data

def unpad_square(data, target_shape):
    shape_difference = [ data.shape[i] - target_shape[i] for i in range(3) ]

    assert (np.array(shape_difference) >= 0).all(), f'Cannot unpad shape from {data.shape} to {target_shape}'
    
    is_odd = lambda x: x % 2 == 1
    any_odd = np.array(list(map(is_odd, shape_difference))).any()
    
    if any_odd:
        print(f'Warning: Input image has a odd numbered dimension {target_shape}. Removing padding will cause unexpected behavior. Skipping...')
        return data

    unpad = [ s // 2 for s in shape_difference ]
    
    data = data[ 
            unpad[0]:data.shape[0] - unpad[0], 
            unpad[1]:data.shape[1] - unpad[1], 
            unpad[2]:data.shape[2] - unpad[2]
    ]
    
    return data