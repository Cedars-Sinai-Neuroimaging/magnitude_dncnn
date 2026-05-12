import argparse
import logging
import nibabel as nib
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import pdb
import tensorflow as tf
from tqdm import tqdm
import yaml

from patchify import patchify, unpatchify

from models import get_model
from utils import np_to_tfdataset, rescale_magnitude, pad_square, unpad_square

PAD_TO_SHAPE = 384 # pad first two dimensions to shape
PATCH_SHAPE = (256, 256, 1)
EXTRACT_STEP = (128, 128, 1)

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    assert args.input.endswith('.nii.gz'), f'Expected nifti image for denoising'

    input_nii = nib.load(args.input)
    input_npy = np.array(input_nii.dataobj)
    
    # preprocess data
    min_before, max_before = np.min(input_npy), np.max(input_npy)
    data = rescale_magnitude(input_npy)
    data = pad_square(data, sag_axis=args.sagittal_axis - 1, pad_to=PAD_TO_SHAPE) # subtract 1 to change to 0-indexing
    patches = patchify(data, PATCH_SHAPE, EXTRACT_STEP)
    patches_shape = patches.shape
    patches = patches.reshape(-1, *PATCH_SHAPE)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = get_model(2, PATCH_SHAPE, load_model_path=args.model)

        patches = np_to_tfdataset(patches)
        print('Denoising input data:', args.input)
        print('Input shape:', input_npy.shape)
        patches = model.predict(
                patches,
                verbose=1,
                batch_size=32
        )
        print('Denoising complete.')
        
    patches = np.squeeze(patches)
    patches = patches.reshape(patches_shape)
    
    # undo preprocessing
    output_npy = unpatchify(patches, data.shape)
    output_npy = unpad_square(output_npy, input_npy.shape)
    output_npy = rescale_magnitude(output_npy, t_min=min_before, t_max=max_before) # rescale back to original
    output_nii = nib.Nifti1Image(output_npy, affine=input_nii.affine, header=input_nii.header)
    print('Saving output to:', args.output)
    nib.save(output_nii, args.output)

    print('Done!')

def create_parser():
    parser = argparse.ArgumentParser()
    example_str = 'python denoise.py input_file.nii.gz output_file.nii.gz model'
    
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('model')
    parser.add_argument('--sagittal_axis', type=int, default=3, choices=[1, 2, 3]) # used to determine axis for padding and patch extraction
    
    return parser

if __name__ == '__main__':
    main()